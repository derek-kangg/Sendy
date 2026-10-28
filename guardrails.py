"""Spend and abuse guardrails for a public app with no login.

One Anthropic key serves every visitor and a single Generate click can issue
up to 60 model calls, so an unprotected deployment is an open wallet. These
limits are in-process and best-effort: they reset when the server restarts
and they are not a substitute for a spend cap in the Anthropic Console —
they make sure one bad afternoon costs dollars, not hundreds.

Every knob is an environment variable so hosting can tune it without a code
change:

    SENDY_DISABLED=1                      kill switch — every AI action refuses
    SENDY_DAILY_BUDGET_USD=5              estimated spend after which AI actions stop
    SENDY_MAX_GENERATIONS_PER_DAY=150     betas generated, across all visitors
    SENDY_MAX_GENERATIONS_PER_VISITOR=8   per visitor (browser session) per day
    SENDY_MAX_DETECTIONS_PER_VISITOR=20
    SENDY_MAX_COACHING_PER_VISITOR=10

Usage is metered by wrapping the Anthropic client (see wrap_client), so the
engine modules stay unaware of this file.
"""

import logging
import os
import threading
import time

log = logging.getLogger("sendy.guardrails")

# Sonnet 4.6 list prices, USD per token
_PRICE = {
    "input":       3.00 / 1e6,
    "output":     15.00 / 1e6,
    "cache_write": 3.75 / 1e6,
    "cache_read":  0.30 / 1e6,
}


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class LimitExceeded(Exception):
    """Raised before an AI action when a guardrail says no. The message is
    written for the climber, not the developer."""


class Ledger:
    """Daily counters and an estimated spend, shared by every session in the
    process. Thread-safe; everything resets at UTC midnight."""

    ACTIONS = ("detect", "generate", "coach")

    def __init__(self):
        self._lock = threading.Lock()
        self._reset_locked()

    def _reset_locked(self):
        self._day = time.strftime("%Y-%m-%d", time.gmtime())
        self._global = {a: 0 for a in self.ACTIONS}
        self._visitor = {}                      # visitor_id -> {action: count}
        self.spent_usd = 0.0
        self.tokens = {k: 0 for k in _PRICE}

    def _roll_day_locked(self):
        if time.strftime("%Y-%m-%d", time.gmtime()) != self._day:
            self._reset_locked()

    # ---- limits -----------------------------------------------------------

    def check(self, action, visitor_id):
        """Raise LimitExceeded if `visitor_id` may not perform `action` now."""
        if os.getenv("SENDY_DISABLED", "").strip().lower() in ("1", "true", "yes"):
            raise LimitExceeded("Sendy's AI features are switched off right now — check back later.")

        with self._lock:
            self._roll_day_locked()
            budget = _env_float("SENDY_DAILY_BUDGET_USD", 5.0)
            if budget > 0 and self.spent_usd >= budget:
                raise LimitExceeded(
                    "Sendy has used up today's AI budget. Come back tomorrow — "
                    "routes you already generated still work."
                )

            mine = self._visitor.setdefault(visitor_id, {a: 0 for a in self.ACTIONS})
            if action == "generate":
                per_visitor = _env_int("SENDY_MAX_GENERATIONS_PER_VISITOR", 8)
                per_day     = _env_int("SENDY_MAX_GENERATIONS_PER_DAY", 150)
                if mine["generate"] >= per_visitor:
                    raise LimitExceeded(
                        f"You've generated {per_visitor} betas today — that's the daily limit per climber. "
                        "Your routes are still here; new betas unlock tomorrow."
                    )
                if self._global["generate"] >= per_day:
                    raise LimitExceeded(
                        "Sendy is busy today and has reached its daily limit for new betas. "
                        "Please try again tomorrow."
                    )
            elif action == "detect":
                per_visitor = _env_int("SENDY_MAX_DETECTIONS_PER_VISITOR", 20)
                if mine["detect"] >= per_visitor:
                    raise LimitExceeded(
                        f"You've run hold detection {per_visitor} times today — that's the daily limit per climber."
                    )
                if self._global["detect"] >= 2 * _env_int("SENDY_MAX_GENERATIONS_PER_DAY", 150):
                    raise LimitExceeded(
                        "Sendy is busy today and has reached its daily limit. Please try again tomorrow."
                    )
            elif action == "coach":
                per_visitor = _env_int("SENDY_MAX_COACHING_PER_VISITOR", 10)
                if mine["coach"] >= per_visitor:
                    raise LimitExceeded(
                        f"You've asked for coaching {per_visitor} times today — that's the daily limit per climber."
                    )

    def count(self, action, visitor_id):
        """Record that `visitor_id` started `action` (call after check passes)."""
        with self._lock:
            self._roll_day_locked()
            self._global[action] = self._global.get(action, 0) + 1
            mine = self._visitor.setdefault(visitor_id, {a: 0 for a in self.ACTIONS})
            mine[action] = mine.get(action, 0) + 1

    # ---- metering ----------------------------------------------------------

    def record_usage(self, usage):
        """Add one API response's token usage to today's estimated spend.
        Returns the estimated cost of that response in USD."""
        if usage is None:
            return 0.0
        got = {
            "input":       getattr(usage, "input_tokens", 0) or 0,
            "output":      getattr(usage, "output_tokens", 0) or 0,
            "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read":  getattr(usage, "cache_read_input_tokens", 0) or 0,
        }
        cost = sum(got[k] * _PRICE[k] for k in got)
        with self._lock:
            self._roll_day_locked()
            for k, v in got.items():
                self.tokens[k] += v
            self.spent_usd += cost
        return cost

    def status(self):
        with self._lock:
            self._roll_day_locked()
            return {
                "day": self._day,
                "spent_usd": round(self.spent_usd, 4),
                "tokens": dict(self.tokens),
                "actions": dict(self._global),
                "visitors": len(self._visitor),
            }


LEDGER = Ledger()


class _MeteredMessages:
    def __init__(self, inner, ledger):
        self._inner = inner
        self._ledger = ledger

    def create(self, **kwargs):
        resp = self._inner.create(**kwargs)
        cost = self._ledger.record_usage(getattr(resp, "usage", None))
        log.info("claude call model=%s cost=$%.4f today=$%.3f",
                 kwargs.get("model"), cost, self._ledger.spent_usd)
        return resp

    def __getattr__(self, name):
        return getattr(self._inner, name)


class MeteredClient:
    """Drop-in stand-in for anthropic.Anthropic that meters every
    messages.create call into a Ledger. Everything else passes through."""

    def __init__(self, client, ledger=None):
        self._client = client
        self.messages = _MeteredMessages(client.messages, ledger or LEDGER)

    def __getattr__(self, name):
        return getattr(self._client, name)


def wrap_client(client, ledger=None):
    return MeteredClient(client, ledger)
