"""The ledger must stop a runaway visitor, a runaway day, and a blown budget,
and the client wrapper must meter every call without changing its result."""

import os
import unittest
from types import SimpleNamespace

import guardrails
from guardrails import Ledger, LimitExceeded, wrap_client


class _Env:
    """Temporarily set environment variables."""

    def __init__(self, **vals):
        self.vals = vals

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.vals}
        for k, v in self.vals.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _usage(**kw):
    base = dict(input_tokens=0, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=0)
    base.update(kw)
    return SimpleNamespace(**base)


class LedgerTests(unittest.TestCase):
    def test_per_visitor_generation_limit(self):
        ledger = Ledger()
        with _Env(SENDY_MAX_GENERATIONS_PER_VISITOR=2, SENDY_DISABLED=None):
            for _ in range(2):
                ledger.check("generate", "alice")
                ledger.count("generate", "alice")
            with self.assertRaises(LimitExceeded) as ctx:
                ledger.check("generate", "alice")
            self.assertIn("2 betas", str(ctx.exception))
            ledger.check("generate", "bob")          # a different visitor is fine

    def test_global_daily_generation_limit(self):
        ledger = Ledger()
        with _Env(SENDY_MAX_GENERATIONS_PER_DAY=3, SENDY_MAX_GENERATIONS_PER_VISITOR=99, SENDY_DISABLED=None):
            for i in range(3):
                ledger.check("generate", f"v{i}")
                ledger.count("generate", f"v{i}")
            with self.assertRaises(LimitExceeded):
                ledger.check("generate", "new-visitor")

    def test_budget_stops_every_action(self):
        ledger = Ledger()
        cost = ledger.record_usage(_usage(input_tokens=1_000_000))   # $3 of input
        self.assertAlmostEqual(cost, 3.0)
        with _Env(SENDY_DAILY_BUDGET_USD=2.5, SENDY_DISABLED=None):
            for action in ("detect", "generate", "coach"):
                with self.assertRaises(LimitExceeded):
                    ledger.check(action, "anyone")
        with _Env(SENDY_DAILY_BUDGET_USD=10, SENDY_DISABLED=None):
            ledger.check("generate", "anyone")

    def test_cache_tokens_are_priced_separately(self):
        ledger = Ledger()
        ledger.record_usage(_usage(cache_creation_input_tokens=1_000_000, cache_read_input_tokens=1_000_000))
        self.assertAlmostEqual(ledger.spent_usd, 3.75 + 0.30)

    def test_kill_switch(self):
        ledger = Ledger()
        with _Env(SENDY_DISABLED="1"):
            with self.assertRaises(LimitExceeded):
                ledger.check("detect", "anyone")
        with _Env(SENDY_DISABLED=None):
            ledger.check("detect", "anyone")

    def test_status_summary(self):
        ledger = Ledger()
        ledger.count("detect", "a")
        ledger.record_usage(_usage(output_tokens=1000))
        s = ledger.status()
        self.assertEqual(s["actions"]["detect"], 1)
        self.assertEqual(s["visitors"], 1)
        self.assertAlmostEqual(s["spent_usd"], 0.015)


class MeteredClientTests(unittest.TestCase):
    def test_wrapper_passes_through_and_meters(self):
        calls = []

        class FakeMessages:
            def create(self, **kw):
                calls.append(kw)
                return SimpleNamespace(content=[], usage=_usage(input_tokens=2000, output_tokens=100))

        fake = SimpleNamespace(messages=FakeMessages(), api_key="k")
        ledger = Ledger()
        client = wrap_client(fake, ledger)
        resp = client.messages.create(model="m", max_tokens=5, messages=[])
        self.assertEqual(calls[0]["model"], "m")
        self.assertEqual(resp.content, [])
        self.assertAlmostEqual(ledger.spent_usd, 2000 * 3e-6 + 100 * 15e-6)
        self.assertEqual(client.api_key, "k")          # other attributes pass through

    def test_default_ledger_is_the_module_singleton(self):
        fake = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(usage=None)))
        client = wrap_client(fake)
        self.assertIs(client.messages._ledger, guardrails.LEDGER)


if __name__ == "__main__":
    unittest.main()
