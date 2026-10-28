"""Headless beta-generation engine.

All the sequence logic lives here, independent of Streamlit, so the same
code drives the web app, the eval harness (scripts/eval_route.py), and a
future API backend. Python owns the body state; the model only ever picks
the next single move, and every move passes physical validation before it
is accepted.

The prompt is split so the expensive part is cached: everything that is
fixed for the whole route (coaching rules, technique library, the hold
list) goes in the system block with cache_control, and only the body
state, history and reachable lists change per call. Input tokens are ~90%
of a route's cost and ~75% of each call's input is that static block.

generate_sequence_iteratively() returns a GenerationResult. It never raises
for API trouble or a model that cannot find a legal move: whatever was
generated so far comes back with a status and a reason the UI can show.
"""

import logging
import math
from dataclasses import dataclass, field

import anthropic

from climbing_knowledge import technique_library

GENERATION_MODEL       = "claude-sonnet-4-6"
MAX_MOVES              = 20
MAX_ATTEMPTS           = 3       # one request + two corrections per move
REQUEST_TIMEOUT_S      = 60.0    # per call; the SDK default is 10 minutes
DEFAULT_WALL_HEIGHT_CM = 400     # photo height ≈ one bouldering wall

HANDS = ("right hand", "left hand", "both hands")
LIMBS = (
    "right hand", "left hand", "both hands",
    "right foot", "left foot", "both feet",
    "flag right foot", "flag left foot",
    "smear right foot", "smear left foot",
    "swap right foot", "swap left foot",
)

log = logging.getLogger("climbai.engine")


@dataclass
class GenerationResult:
    """What the engine produced, complete or not.

    status: complete   — both hands finished on the finish hold
            incomplete — hit the move limit before the finish
            stalled    — the model could not propose a legal move
            error      — the API failed mid-route (reason is user-facing)
    """
    sequence: list
    states: list
    status: str = "complete"
    reason: str = ""
    calls: int = 0
    finish_hold: int = None

    @property
    def ok(self):
        return self.status == "complete"

    def __iter__(self):
        # Backward compatible with `sequence, states = generate_sequence_iteratively(...)`
        return iter((self.sequence, self.states))


def build_holds_description(holds):
    if not holds:
        return "No holds detected."
    sorted_holds = sorted(holds, key=lambda h: h["y"], reverse=True)
    all_x = [h["x"] for h in holds]
    mid_x = (max(all_x) + min(all_x)) / 2

    lines = ["Holds listed bottom to top:"]
    for h in sorted_holds:
        side      = "LEFT" if h["x"] < mid_x else "RIGHT"
        hold_type = h.get("hold_type", "unknown")
        best_use  = h.get("best_use", "both")
        size      = h.get("size", "unknown")
        orient    = h.get("orientation", "unknown")
        orient_str = f", orientation={orient}" if orient not in ("unknown", "top") else ""
        lines.append(
            f"Hold {h['number']}: {hold_type}, {size}, {side} side, best_use={best_use}{orient_str}, x={h['x']}, y={h['y']}"
        )
    return "\n".join(lines)


def apply_limb_to_state(new_state, limb, hold):
    """Apply a single limb placement to state dict in place."""
    if limb == "right hand":
        new_state["RH"] = hold
    elif limb == "left hand":
        new_state["LH"] = hold
    elif limb == "both hands":
        new_state["RH"] = hold
        new_state["LH"] = hold
    elif limb == "right foot":
        new_state["RF"] = hold
        if hold is not None and new_state["LF"] == hold:
            new_state["LF"] = None
    elif limb == "left foot":
        new_state["LF"] = hold
        if hold is not None and new_state["RF"] == hold:
            new_state["RF"] = None
    elif limb == "both feet":
        new_state["RF"] = hold
        new_state["LF"] = hold
    elif limb == "swap right foot":
        new_state["RF"] = hold
        new_state["LF"] = None
    elif limb == "swap left foot":
        new_state["LF"] = hold
        new_state["RF"] = None
    elif limb in ("smear right foot", "flag right foot"):
        new_state["RF"] = None
    elif limb in ("smear left foot", "flag left foot"):
        new_state["LF"] = None


def apply_move_to_state(state, move):
    """Apply a move to the body state dict."""
    new_state = state.copy()
    apply_limb_to_state(new_state, move["limb"], move["hold"])
    return new_state


def format_state(state):
    def fmt(v):
        return f"Hold {v}" if v is not None else "wall/air"
    return f"LH: {fmt(state['LH'])} | RH: {fmt(state['RH'])} | LF: {fmt(state['LF'])} | RF: {fmt(state['RF'])}"


def _hold_str(hold):
    return f"Hold {hold}" if hold is not None else "wall"


def _dist(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _foot_options(state, hold_map, max_span_px):
    """Holds a foot could move to right now: not under a hand, no higher than
    the higher hand (a small margin allows heel hooks at hand height), and
    within body extension of the nearer hand. Used for both the prompt and
    validation so the model is only ever offered legal footholds."""
    hand_holds = [hold_map[state[h]] for h in ("LH", "RH") if state[h] in hold_map]
    if not hand_holds:
        return []
    top_y  = min(h["y"] for h in hand_holds)
    margin = 0.15 * max_span_px if max_span_px else 30
    held   = {state["LH"], state["RH"]}
    out = []
    for h in hold_map.values():
        if h["number"] in held:
            continue
        if h["y"] < top_y - margin:
            continue
        if max_span_px and min(_dist(h, hh) for hh in hand_holds) > max_span_px:
            continue
        out.append(h)
    out.sort(key=lambda h: h["y"])
    return out


def friendly_api_error(exc):
    """A one-line, climber-facing explanation of an Anthropic SDK exception."""
    if isinstance(exc, anthropic.AuthenticationError):
        return "The AI service rejected the app's API key — the owner needs to fix this."
    if isinstance(exc, anthropic.RateLimitError):
        return "The AI service is busy right now — wait a minute and try again."
    if isinstance(exc, anthropic.APITimeoutError):
        return "The AI service took too long to answer — try again."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Couldn't reach the AI service — check your connection and try again."
    if isinstance(exc, anthropic.APIStatusError):
        return f"The AI service returned an error ({exc.status_code}) — try again in a moment."
    return "Something went wrong while generating the beta — try again."


def generate_sequence_iteratively(
    client, hold_descriptions, holds, graph, start_holds,
    height_cm, difficulty, wall_angle, finish_style, extra_notes,
    image_height=None, progress_callback=None,
    wall_height_cm=DEFAULT_WALL_HEIGHT_CM, finish_hold=None,
):
    """
    Generate a route sequence one move at a time.
    Python tracks body state; the model only picks the next single move.
    `client` is an anthropic.Anthropic instance (or anything with the same
    messages.create). Returns a GenerationResult.
    """
    if not start_holds:
        raise ValueError("At least one start hold is required.")

    hold_map = {h["number"]: h for h in holds}
    if finish_hold in hold_map:
        finish_hold_num = finish_hold
    else:
        finish_hold_num = min(holds, key=lambda h: h["y"])["number"]

    # Full body extension (feet to hand) in pixels — same wall-scale
    # assumption route_graph uses. Guards against endless hand moves with
    # planted feet, which no human can do.
    if image_height and wall_height_cm:
        pixels_per_cm = image_height / float(wall_height_cm)
        max_span_px = height_cm * 1.30 * pixels_per_cm
    else:
        max_span_px = None

    # --- start position -----------------------------------------------------
    # Two-hand starts: the left-most hold is the left hand, whatever order
    # the climber tapped them in.
    start_holds = sorted(start_holds[:2], key=lambda h: h["x"])
    start_hand_nums = {h["number"] for h in start_holds}
    start_x = sum(h["x"] for h in start_holds) / len(start_holds)
    start_y = max(h["y"] for h in start_holds)          # y of the lower start hand

    # Starting feet: the lowest holds below the hands, within reach of them.
    # Nothing below the hands means a sit start — feet smear.
    foot_candidates = [
        h for h in holds
        if h["y"] > start_y and h["number"] not in start_hand_nums
        and (not max_span_px or min(_dist(h, s) for s in start_holds) <= max_span_px)
    ]
    foot_candidates.sort(key=lambda h: (-h["y"], abs(h["x"] - start_x)))
    start_feet = sorted(foot_candidates[:2], key=lambda h: h["x"])
    lf_hold = start_feet[0]["number"] if len(start_feet) > 0 else None
    rf_hold = start_feet[1]["number"] if len(start_feet) > 1 else None

    def foot_cue():
        if lf_hold is None and rf_hold is None:
            return "feet smear on the wall (sit start if the holds are low)"
        return f"LF {_hold_str(lf_hold) if lf_hold else 'smear'}, RF {_hold_str(rf_hold) if rf_hold else 'smear'}"

    if len(start_holds) == 1:
        s = start_holds[0]["number"]
        state = {"LH": s, "RH": s, "LF": lf_hold, "RF": rf_hold}
        start_move = {"move_number": 0, "limb": "both hands", "hold": s, "action": "start",
                      "cue": f"Both hands Hold {s}, {foot_cue()}"}
    else:
        l, r = start_holds[0]["number"], start_holds[1]["number"]
        state = {"LH": l, "RH": r, "LF": lf_hold, "RF": rf_hold}
        start_move = {"move_number": 0, "limb": "both hands", "hold": None, "action": "start",
                      "cue": f"LH Hold {l}, RH Hold {r}, {foot_cue()}"}

    sequence = [start_move]
    states = [dict(state)]
    result = GenerationResult(sequence, states, finish_hold=finish_hold_num)

    # --- static prompt (cached for the whole route) ---------------------------
    angle_context = {
        "Slab (less than vertical)": "slab — trust feet, stand tall, smearing common",
        "Vertical": "vertical — balance arm and leg use equally",
        "Slight overhang": "slight overhang — hips close to wall, feet still important",
        "Steep overhang": "steep overhang — body tension critical, move efficiently",
        "Roof": "roof — heel hooks, toe hooks, core tension essential",
    }
    finish_context = {
        "TOP label":                    "marked TOP",
        "Tape on hold":                 "marked with tape",
        "Top-out (both hands on top)":  "top-out, both hands on wall top",
        "Not marked":                   "highest hold of this colour",
    }
    extra = f"\nExtra notes: {extra_notes}" if (extra_notes or "").strip() else ""

    hold_positions = "\n".join(
        f"Hold {h['number']}: x={h['x']}, y={h['y']}, size={h.get('size', '?')}, type={h.get('hold_type', '?')}, use={h.get('best_use', 'both')}"
        + (f", ORIENTATION={h['orientation']}" if h.get("orientation") not in (None, "unknown", "top") else "")
        for h in sorted(holds, key=lambda x: x["y"], reverse=True)
    )

    system_text = f"""You are an experienced climbing coach generating a bouldering sequence one move at a time. Each turn you receive the climber's current body state and must submit the single best next move with the submit_move tool.

ROUTE
FINISH: Hold {finish_hold_num} ({finish_context.get(finish_style, finish_style)})
WALL: {angle_context.get(wall_angle, wall_angle)}
CLIMBER: {height_cm}cm, {difficulty}{extra}

HOLD INFO (type, size, position — y grows downward, so a smaller y is higher on the wall):
{hold_positions}

MECHANICS:
- One limb moves per step — always.
- Hands always go to a hold. Never smear or flag a hand.
- Two hands CAN share a hold (match)
- Two feet CANNOT share a small chip — foot swap instead
- Only move feet to holds in the FOOTHOLDS list you are given, or smear/flag on the wall (hold = null)
- As your hands climb, your feet MUST follow — a hand hold beyond full body extension from your feet is unreachable and will be rejected

FOOT ECONOMY — feet are support, hands drive progress:
- HANDS make upward progress. Move a hand on most moves.
- Only move a foot when the current foot position genuinely cannot support the next hand reach.
- Do NOT move feet proactively or to "prepare" — if feet are stable, leave them and move a hand instead.
- A pivot foothold (a good mid-height chip) should be reused via swaps as hands climb — do not abandon it for a new one each move.
- Flag the free foot when one foot is anchored on the pivot — flagging is stable, not a last resort.
- If in doubt between a foot move and a hand move: move the hand.

{technique_library(difficulty, wall_angle)}
Only move hands to holds shown in the reachable lists you are given.
When a hand reaches Hold {finish_hold_num}, output action=finish."""

    move_tool = {
        "name": "submit_move",
        "description": "Submit the single best next climbing move given the current body state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limb": {"type": "string", "enum": list(LIMBS)},
                "hold": {
                    "type": ["integer", "null"],
                    "description": "Hold number, or null for smear/flag against wall"
                },
                "action": {
                    "type": "string",
                    "enum": ["move", "match", "flag", "smear", "swap", "heel hook", "toe hook", "drop knee",
                             "knee bar", "backstep", "rockover", "deadpoint", "dyno", "mantle", "finish"]
                },
                "cue": {"type": "string", "description": "One short technique cue under 10 words"},
            },
            "required": ["limb", "hold", "action", "cue"],
        },
    }

    def reachable_str(hold_num):
        if hold_num is None:
            return "none"
        entries = graph.get(hold_num, [])
        if not entries:
            return "none in reach"
        return ", ".join(f"Hold {r['hold']} ({r['difficulty']}, {r['direction']})" for r in entries[:8])

    def footholds_str():
        opts = _foot_options(state, hold_map, max_span_px)
        if not opts:
            return "none — smear or flag"
        return ", ".join(f"Hold {h['number']} ({h.get('size', '?')})" for h in opts[:10])

    def request_move(user_prompt, correction=None, previous=None):
        messages = [{"role": "user", "content": user_prompt}]
        if correction:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": "I'll reconsider."}]})
            prev = f" Your proposed move ({previous.get('limb')} -> {_hold_str(previous.get('hold'))}) is invalid:" if previous else ""
            messages.append({"role": "user", "content": f"CORRECTION NEEDED:{prev} {correction} Submit a different move."})
        resp = client.messages.create(
            model=GENERATION_MODEL,
            max_tokens=256,
            timeout=REQUEST_TIMEOUT_S,
            system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
            tools=[move_tool],
            tool_choice={"type": "any"},
            messages=messages,
        )
        result.calls += 1
        for block in resp.content:
            if block.type == "tool_use" and block.name == "submit_move":
                return dict(block.input)
        return None

    def validate_move(m):
        """Returns an error string if the move violates a physical rule, else None."""
        if m is None:
            return "No move was submitted."
        limb = m.get("limb", "")
        hold = m.get("hold")
        if limb not in LIMBS:
            return f"'{limb}' is not a valid limb."
        if hold is not None and not isinstance(hold, int):
            return "hold must be a hold number or null."
        if hold is not None and hold not in hold_map:
            return f"Hold {hold} does not exist on this wall. Choose a valid hold number."

        if limb in HANDS:
            if hold is None:
                return "Hands must go to a hold — use a foot limb to smear or flag."
            if limb == "right hand":
                if state["RH"] == hold:
                    return f"RH is already on Hold {hold} — choose a different move."
                if hold not in [r["hold"] for r in graph.get(state["RH"], [])]:
                    return f"Hold {hold} is not reachable from RH (Hold {state['RH']}). Choose a hold from the reachable list."
            elif limb == "left hand":
                if state["LH"] == hold:
                    return f"LH is already on Hold {hold} — choose a different move."
                if hold not in [r["hold"] for r in graph.get(state["LH"], [])]:
                    return f"Hold {hold} is not reachable from LH (Hold {state['LH']}). Choose a hold from the reachable list."
            else:  # both hands
                reach = {r["hold"] for r in graph.get(state["LH"], [])} | {r["hold"] for r in graph.get(state["RH"], [])}
                reach |= {state["LH"], state["RH"]}
                if hold not in reach:
                    return f"Hold {hold} is not reachable from either hand. Choose a hold from the reachable lists."

            # Over-extension: a hand cannot end up beyond full body extension
            # from the feet — at some point a foot has to move up
            if max_span_px:
                foot_holds = [state[f] for f in ("LF", "RF") if state[f] is not None and state[f] in hold_map]
                if foot_holds:
                    closest_foot = min(_dist(hold_map[f], hold_map[hold]) for f in foot_holds)
                    if closest_foot > max_span_px:
                        return (f"Hold {hold} is beyond full body extension from your current feet. "
                                f"Move a foot up to a higher foothold first, then reach with the hand.")
            return None

        # ---- feet ----
        if limb.startswith(("flag", "smear")):
            if hold is not None:
                return "Flag and smear moves go on the wall — set hold to null."
            return None

        if hold is None:
            return "A foot move needs a hold number; use a smear or flag limb for the wall."

        moving = "RF" if "right" in limb else "LF"
        other  = "LF" if moving == "RF" else "RF"
        if limb == "both feet":
            if hold_map[hold].get("size", "").startswith("small"):
                return f"Both feet cannot share Hold {hold} — it is a small chip. Move one foot at a time."
        else:
            if state[moving] == hold:
                return f"{moving} is already on Hold {hold} — choose a different move."
            if state[other] == hold and hold_map[hold].get("size", "").startswith("small") \
                    and not limb.startswith("swap"):
                return f"Both feet cannot share Hold {hold} — it is a small chip. Use foot swap or choose a different hold."

        options = {h["number"] for h in _foot_options(state, hold_map, max_span_px)}
        if hold not in options:
            return (f"Hold {hold} is not a usable foothold from this position — it is above your hands, "
                    f"under a hand, or too far from them. Choose from the FOOTHOLDS list.")
        return None

    # --- the loop --------------------------------------------------------------
    for move_num in range(1, MAX_MOVES + 1):
        if progress_callback:
            progress_callback(move_num, MAX_MOVES)

        history = "\n".join(
            f"  {m['move_number']}: {m['limb']} -> {_hold_str(m['hold'])} ({m['action']}) — {m.get('cue', '')}"
            for m in sequence
        )
        user_prompt = f"""CURRENT BODY STATE:
{format_state(state)}

MOVE HISTORY:
{history}

REACHABLE FROM CURRENT HANDS:
From LH (Hold {state['LH']}): {reachable_str(state['LH'])}
From RH (Hold {state['RH']}): {reachable_str(state['RH'])}

FOOTHOLDS you can move a foot to right now (below your hands, within reach):
{footholds_str()}

What is the single best next move?"""

        move, error = None, None
        try:
            for attempt in range(MAX_ATTEMPTS):
                move = request_move(user_prompt, correction=error if attempt else None, previous=move)
                error = validate_move(move)
                if error is None:
                    break
        except Exception as exc:
            log.exception("API failure at move %d after %d calls", move_num, result.calls)
            result.status = "error"
            result.reason = friendly_api_error(exc)
            break

        if error is not None:
            log.warning("move %d stalled after %d attempts: %s", move_num, MAX_ATTEMPTS, error)
            result.status = "stalled"
            result.reason = f"Couldn't find a legal move {move_num}: {error}"
            break

        move["move_number"] = move_num
        state = apply_move_to_state(state, move)
        sequence.append(move)
        states.append(dict(state))

        if move["action"] == "finish":
            break
        if move["hold"] == finish_hold_num and move["limb"] in HANDS:
            break

    # A boulder is finished with BOTH hands matched on the finish hold —
    # if only one hand got there, append the match deterministically.
    lh_on = state["LH"] == finish_hold_num
    rh_on = state["RH"] == finish_hold_num
    if lh_on != rh_on:
        match_move = {
            "move_number": sequence[-1]["move_number"] + 1,
            "limb": "left hand" if rh_on else "right hand",
            "hold": finish_hold_num,
            "action": "match",
            "cue": "Match hands on the finish hold and control it",
        }
        state = apply_move_to_state(state, match_move)
        sequence.append(match_move)
        states.append(dict(state))
        lh_on = rh_on = True

    if lh_on and rh_on:
        result.status = "complete"
        result.reason = ""
    elif result.status == "complete":          # loop simply ran out of moves
        result.status = "incomplete"
        result.reason = f"Reached the {MAX_MOVES}-move limit before the finish hold."

    log.info("generation %s: %d moves, %d calls%s", result.status, len(sequence) - 1, result.calls,
             f" — {result.reason}" if result.reason else "")
    return result
