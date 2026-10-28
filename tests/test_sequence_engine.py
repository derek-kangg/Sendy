"""The engine must never accept an illegal move, never lose paid-for moves on
failure, and always say how it ended. Uses a scripted fake client against
the real route-01 holds — no API calls."""

import json
import os
import unittest
from types import SimpleNamespace

import anthropic
import httpx

import sequence_engine
from route_graph import build_reachability_graph
from sequence_engine import GenerationResult, generate_sequence_iteratively

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "test_routes", "route-01-black-slab", "route_data.json"), encoding="utf-8") as f:
    ROUTE = json.load(f)
HOLDS = ROUTE["holds"]
IMG_W, IMG_H = 770, 1024
HEIGHT_CM = 170


def _connection_error():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(request=req)


class ScriptedClient:
    """Returns the scripted moves in order; an Exception instance in the
    script is raised instead. Cycles when `cycle` is set; otherwise running
    out of script behaves like a dropped connection, so tests that only care
    about the first moves get a partial result with status='error' and tests
    that assert status='complete' must script the whole route."""

    def __init__(self, script, cycle=False):
        self.script = list(script)
        self.cycle = cycle
        self.calls = []
        self.messages = self
        self._i = 0

    def create(self, **kw):
        self.calls.append(kw)
        if self._i >= len(self.script):
            if not self.cycle:
                raise _connection_error()
            self._i = 0
        item = self.script[self._i]
        self._i += 1
        if isinstance(item, Exception):
            raise item
        block = SimpleNamespace(type="tool_use", name="submit_move", input=dict(item))
        return SimpleNamespace(content=[block], stop_reason="tool_use", usage=None)


def mv(limb, hold, action="move", cue="go"):
    return {"limb": limb, "hold": hold, "action": action, "cue": cue}


def run(client, start=(5,), **kw):
    graph = build_reachability_graph(HOLDS, IMG_H, IMG_W, HEIGHT_CM)
    hold_map = {h["number"]: h for h in HOLDS}
    start_holds = [hold_map[n] for n in start]
    return generate_sequence_iteratively(
        client, "", HOLDS, graph, start_holds,
        HEIGHT_CM, "Intermediate", "Vertical", "TOP label", "",
        image_height=IMG_H, **kw,
    )


class StartPositionTests(unittest.TestCase):
    def test_feet_auto_placed_on_lowest_holds_below_hands_left_to_right(self):
        res = run(ScriptedClient([mv("right hand", 6)]))
        self.assertEqual(res.states[0], {"LH": 5, "RH": 5, "LF": 2, "RF": 1})
        self.assertIn("LF Hold 2, RF Hold 1", res.sequence[0]["cue"])

    def test_two_hand_start_assigns_left_hand_to_leftmost_hold_regardless_of_tap_order(self):
        res = run(ScriptedClient([mv("right hand", 7)]), start=(6, 5))   # tapped right-to-left
        self.assertEqual(res.states[0]["LH"], 5)
        self.assertEqual(res.states[0]["RH"], 6)

    def test_sit_start_has_no_hold_none_in_cue(self):
        res = run(ScriptedClient([mv("right hand", 6)]), start=(2,))   # lowest hold -> nothing below
        self.assertIsNone(res.states[0]["LF"])
        self.assertNotIn("None", res.sequence[0]["cue"])
        self.assertIn("smear", res.sequence[0]["cue"])

    def test_no_start_holds_is_a_programming_error(self):
        with self.assertRaises(ValueError):
            run(ScriptedClient([]), start=())


class ValidationTests(unittest.TestCase):
    def test_hand_to_wall_is_rejected_then_corrected(self):
        client = ScriptedClient([
            mv("left hand", None), mv("right hand", 6), mv("left hand", 7), mv("right foot", 3),
            mv("right hand", 8), mv("left foot", 6), mv("left hand", 9, "finish"),
        ])
        res = run(client)
        self.assertEqual(res.status, "complete", res.reason)
        self.assertEqual([m["hold"] for m in res.sequence[1:]], [6, 7, 3, 8, 6, 9, 9])   # the None move never landed
        self.assertIn("CORRECTION NEEDED", client.calls[1]["messages"][-1]["content"])
        self.assertIn("Hands must go to a hold", client.calls[1]["messages"][-1]["content"])

    def test_unreachable_hand_move_never_accepted_even_on_third_attempt(self):
        client = ScriptedClient([mv("right hand", 4)] * 3)                  # 4 is too far sideways
        res = run(client)
        self.assertEqual(res.status, "stalled")
        self.assertEqual(len(res.sequence), 1)                              # only the start
        self.assertEqual(res.calls, sequence_engine.MAX_ATTEMPTS)
        self.assertIn("not reachable", res.reason)

    def test_over_extension_requires_a_foot_move_first(self):
        # From hold 6 the jug (9) is within arm reach but ~840 px from the feet
        client = ScriptedClient([
            mv("right hand", 6), mv("right hand", 9), mv("left hand", 7), mv("right foot", 3),
            mv("right hand", 8), mv("left foot", 6), mv("left hand", 9, "finish"),
        ])
        res = run(client)
        self.assertEqual(res.status, "complete", res.reason)
        self.assertEqual([m["hold"] for m in res.sequence[1:3]], [6, 7])
        self.assertIn("beyond full body extension", client.calls[2]["messages"][-1]["content"])

    def test_foot_above_hands_is_rejected(self):
        client = ScriptedClient([mv("left foot", 7), mv("left foot", 3), mv("right hand", 6)])
        res = run(client)
        self.assertEqual(res.sequence[1]["hold"], 3)
        self.assertEqual(res.states[1]["LF"], 3)
        self.assertIn("not a usable foothold", client.calls[1]["messages"][-1]["content"])

    def test_same_hand_to_same_hold_is_rejected(self):
        client = ScriptedClient([mv("right hand", 5), mv("right hand", 6)])
        res = run(client)
        self.assertEqual(res.sequence[1]["hold"], 6)
        self.assertIn("already on Hold 5", client.calls[1]["messages"][-1]["content"])

    def test_two_feet_on_small_chip_rejected_but_swap_allowed(self):
        # start feet: LF 2, RF 1 (1 and 3 are small chips)
        client = ScriptedClient([mv("left foot", 1), mv("swap left foot", 1, "swap"), mv("right hand", 6)])
        res = run(client)
        self.assertEqual(res.sequence[1]["limb"], "swap left foot")
        self.assertEqual(res.states[1], {"LH": 5, "RH": 5, "LF": 1, "RF": None})
        self.assertIn("small chip", client.calls[1]["messages"][-1]["content"])

    def test_flag_with_a_hold_number_is_rejected(self):
        client = ScriptedClient([mv("flag right foot", 3, "flag"), mv("flag right foot", None, "flag"), mv("right hand", 6)])
        res = run(client)
        self.assertEqual(res.sequence[1]["limb"], "flag right foot")
        self.assertIsNone(res.states[1]["RF"])


class OutcomeTests(unittest.TestCase):
    def test_complete_route_gets_hands_matched_on_finish(self):
        client = ScriptedClient([
            mv("right hand", 6), mv("left hand", 7), mv("right foot", 3),
            mv("right hand", 8), mv("left foot", 6), mv("left hand", 9, "finish"),
        ])
        res = run(client)
        self.assertEqual(res.status, "complete", res.reason)
        self.assertTrue(res.ok)
        self.assertEqual(res.states[-1]["LH"], 9)
        self.assertEqual(res.states[-1]["RH"], 9)
        self.assertEqual(res.sequence[-1]["action"], "match")
        self.assertEqual(res.sequence[-1]["limb"], "right hand")
        self.assertEqual(len(res.sequence), 8)
        self.assertEqual(res.finish_hold, 9)
        self.assertEqual(res.calls, 6)

    def test_api_error_mid_route_keeps_earlier_moves(self):
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        boom = anthropic.RateLimitError("busy", response=httpx.Response(429, request=req), body=None)
        client = ScriptedClient([mv("right hand", 6), mv("left hand", 7), boom])
        res = run(client)
        self.assertEqual(res.status, "error")
        self.assertEqual([m["hold"] for m in res.sequence[1:]], [6, 7])
        self.assertEqual(len(res.states), 3)
        self.assertIn("busy", res.reason)
        self.assertNotIn("Traceback", res.reason)

    def test_move_limit_reports_incomplete(self):
        client = ScriptedClient([mv("right foot", 3), mv("right foot", 1)], cycle=True)
        res = run(client)
        self.assertEqual(res.status, "incomplete")
        self.assertEqual(len(res.sequence), sequence_engine.MAX_MOVES + 1)
        self.assertIn("move limit", res.reason)

    def test_explicit_finish_hold_overrides_highest(self):
        client = ScriptedClient([mv("right hand", 6), mv("left hand", 7, "finish")])
        res = run(client, finish_hold=7)
        self.assertEqual(res.status, "complete", res.reason)
        self.assertEqual(res.finish_hold, 7)
        self.assertEqual(res.states[-1], {"LH": 7, "RH": 7, "LF": 2, "RF": 1})

    def test_taller_wall_scale_shrinks_reach(self):
        # On an 8 m wall the same pixels are twice the distance: hold 6 is now
        # fine from 5, but 7 from 5 (334 px ≈ 5.2 m) is out of reach.
        graph = build_reachability_graph(HOLDS, IMG_H, IMG_W, HEIGHT_CM, wall_height_cm=800)
        self.assertIn(6, [r["hold"] for r in graph[5]])
        self.assertNotIn(7, [r["hold"] for r in graph[5]])

    def test_result_unpacks_like_the_old_tuple(self):
        res = run(ScriptedClient([mv("right hand", 6)]))
        sequence, states = res
        self.assertIs(sequence, res.sequence)
        self.assertIs(states, res.states)
        self.assertIsInstance(res, GenerationResult)


class PromptShapeTests(unittest.TestCase):
    def test_static_block_is_cached_and_volatile_block_is_small(self):
        client = ScriptedClient([mv("right hand", 6), mv("left hand", 7)])
        run(client)
        kw = client.calls[0]
        self.assertEqual(kw["timeout"], sequence_engine.REQUEST_TIMEOUT_S)
        self.assertEqual(kw["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertIn("HOLD INFO", kw["system"][0]["text"])
        self.assertIn("GRIP TYPES", kw["system"][0]["text"])
        user = kw["messages"][0]["content"]
        self.assertIn("CURRENT BODY STATE", user)
        self.assertIn("FOOTHOLDS", user)
        self.assertNotIn("HOLD INFO", user)
        self.assertNotIn("GRIP TYPES", user)
        # the static block is identical call to call, so the cache can hit
        self.assertEqual(kw["system"], client.calls[1]["system"])
        self.assertEqual(kw["tools"], client.calls[1]["tools"])

    def test_foothold_list_only_offers_legal_holds(self):
        client = ScriptedClient([mv("right hand", 6)])
        run(client)
        user = client.calls[0]["messages"][0]["content"]
        footholds = user.split("FOOTHOLDS")[1].split("\n")[1]
        # hands on 5 (y=660): 1, 2, 3, 4 are below; 6, 7, 8, 9 are above
        for n in (1, 2, 3, 4):
            self.assertIn(f"Hold {n} ", footholds)
        for n in (6, 7, 8, 9):
            self.assertNotIn(f"Hold {n} ", footholds)


if __name__ == "__main__":
    unittest.main()
