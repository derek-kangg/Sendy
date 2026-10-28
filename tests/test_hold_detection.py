"""Detection failure modes must surface as typed, user-facing errors instead
of empty results or crashes, and the OpenCV colour stage must find the holds
real gym photos taught us it missed. Uses synthetic images and a fake client
— no API calls."""

import os
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

import hold_detection
from hold_detection import (DetectionError, NoHoldsFound, detect_and_validate_holds,
                            find_hold_candidates)


BLUE = (220, 90, 20)   # BGR


def _wall(path, blobs=(), size=(800, 1000), bgr=(60, 60, 60)):
    """Grey wall with filled blue circles at the given (x, y, r) positions."""
    img = np.full((size[1], size[0], 3), bgr, dtype=np.uint8)
    for x, y, r in blobs:
        cv2.circle(img, (x, y), r, BLUE, -1)
    cv2.imwrite(path, img)
    return path


def _covers(cands, x, y):
    """True if some candidate's box contains the point."""
    return any(c["bx"] <= x <= c["bx"] + c["bw"] and c["by"] <= y <= c["by"] + c["bh"] for c in cands)


class _FakeClient:
    def __init__(self, holds=None, stop_reason="tool_use", raise_exc=None):
        self.holds = holds
        self.stop_reason = stop_reason
        self.raise_exc = raise_exc
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.raise_exc:
            raise self.raise_exc
        content = []
        if self.holds is not None:
            content.append(SimpleNamespace(type="tool_use", name="submit_validated_holds",
                                           input={"holds": self.holds}))
        return SimpleNamespace(content=content, stop_reason=self.stop_reason, usage=None)


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "wall.jpg")

    def tearDown(self):
        self.tmp.cleanup()

    def test_candidates_found_on_synthetic_wall(self):
        _wall(self.path, blobs=[(200, 800, 30), (400, 500, 25), (600, 200, 35)])
        cands = find_hold_candidates(self.path, "Blue")
        self.assertEqual(len(cands), 3)
        for c in cands:
            self.assertIn(c["size"], ("small", "medium", "large"))

    def test_unreadable_file_is_a_detection_error_not_a_crash(self):
        with self.assertRaises(DetectionError):
            find_hold_candidates(os.path.join(self.tmp.name, "missing.jpg"), "Blue")

    def test_no_holds_raises_before_any_api_call(self):
        _wall(self.path)                                   # plain wall, nothing blue
        client = _FakeClient(holds=[])
        with self.assertRaises(NoHoldsFound) as ctx:
            detect_and_validate_holds(client, self.path, "Blue")
        self.assertEqual(client.calls, [])
        self.assertIn("blue", str(ctx.exception).lower())

    def test_labels_are_applied_and_tape_dropped(self):
        _wall(self.path, blobs=[(200, 800, 30), (400, 500, 25), (600, 200, 35)])
        client = _FakeClient(holds=[
            {"number": 1, "hold_type": "jug",   "best_use": "handhold", "orientation": "top", "same_hold_as": None},
            {"number": 2, "hold_type": "tape",  "best_use": "both",     "orientation": "unknown", "same_hold_as": None},
            {"number": 3, "hold_type": "crimp", "best_use": "handhold", "orientation": "side-pull left", "same_hold_as": None},
        ])
        image, holds = detect_and_validate_holds(client, self.path, "Blue")
        self.assertEqual(len(holds), 2)
        self.assertEqual([h["number"] for h in holds], [1, 2])      # renumbered
        self.assertEqual({h["hold_type"] for h in holds}, {"jug", "crimp"})
        self.assertEqual(image.size, (800, 1000))
        kw = client.calls[0]
        self.assertGreaterEqual(kw["max_tokens"], 300 + 70 * 3)
        self.assertEqual(kw["timeout"], hold_detection.REQUEST_TIMEOUT_S)

    def test_truncated_empty_response_is_reported(self):
        _wall(self.path, blobs=[(200, 800, 30)])
        client = _FakeClient(holds=[], stop_reason="max_tokens")
        with self.assertRaises(DetectionError) as ctx:
            detect_and_validate_holds(client, self.path, "Blue")
        self.assertIn("cut short", str(ctx.exception))

    def test_truncated_partial_response_still_returns_holds(self):
        _wall(self.path, blobs=[(200, 800, 30), (400, 500, 25)])
        client = _FakeClient(holds=[{"number": 1, "hold_type": "jug", "best_use": "handhold",
                                     "orientation": "top", "same_hold_as": None}],
                             stop_reason="max_tokens")
        _, holds = detect_and_validate_holds(client, self.path, "Blue")
        self.assertEqual(len(holds), 2)
        self.assertEqual(holds[1]["hold_type"], "unknown")   # unlabelled candidate kept as a hold

    def test_candidate_cap(self):
        blobs = [(40 + (i % 10) * 75, 100 + (i // 10) * 110, 14) for i in range(70)]
        _wall(self.path, blobs=blobs)
        self.assertGreater(len(find_hold_candidates(self.path, "Blue")), hold_detection.MAX_CANDIDATES)
        client = _FakeClient(holds=[])
        with self.assertRaises(DetectionError):      # empty validation -> error, but the call was capped
            detect_and_validate_holds(client, self.path, "Blue")
        text = client.calls[0]["messages"][0]["content"][-1]["text"]
        self.assertIn(f"{hold_detection.MAX_CANDIDATES} numbered candidate", text)


class ColourStageTests(unittest.TestCase):
    """The OpenCV candidate stage on synthetic walls: colour ranges,
    wall-paint rejection and the tape trimmer. PNG keeps the colours exact."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "wall.png")

    def tearDown(self):
        self.tmp.cleanup()

    def test_red_hue_wrap(self):
        # Crimson sits at the top of the hue circle (OpenCV hue ~174), where a
        # single 0-10 range never looked; pure red (hue 0) must still count.
        crimson, pure_red = (40, 0, 200), (0, 0, 220)
        self.assertGreaterEqual(cv2.cvtColor(np.uint8([[crimson]]), cv2.COLOR_BGR2HSV)[0, 0, 0], 170)
        img = np.full((1000, 800, 3), (60, 60, 60), dtype=np.uint8)
        holds = [((200, 800), crimson), ((500, 500), crimson), ((650, 200), pure_red)]
        for (x, y), bgr in holds:
            cv2.circle(img, (x, y), 30, bgr, -1)
        cv2.imwrite(self.path, img)
        cands = find_hold_candidates(self.path, "Red")
        self.assertEqual(len(cands), 3)
        for (x, y), _ in holds:
            self.assertTrue(_covers(cands, x, y), f"no red candidate at {(x, y)}")

    def test_white_holds_on_a_light_wall(self):
        # A white-painted wall must not come back as a giant candidate, while
        # the cream and glossy-white holds sitting on it must — and a hold of
        # another colour must not sprout white junk around it.
        paint, cream, gloss = (192, 200, 215), (196, 214, 238), (250, 250, 250)
        img = np.full((1000, 800, 3), paint, dtype=np.uint8)
        holds = [((200, 300), cream), ((550, 450), cream), ((400, 750), gloss)]
        for (x, y), bgr in holds:
            cv2.circle(img, (x, y), 32, bgr, -1)
        cv2.circle(img, (650, 150), 40, BLUE, -1)
        cv2.imwrite(self.path, img)
        cands = find_hold_candidates(self.path, "White")
        for c in cands:
            self.assertLess(c["bw"] * c["bh"], 0.05 * 800 * 1000, f"wall paint came back as a candidate: {c}")
        for (x, y), _ in holds:
            self.assertTrue(_covers(cands, x, y), f"white hold at {(x, y)} not found")
        self.assertFalse(_covers(cands, 650, 150), "the blue hold is not a white candidate")
        self.assertEqual(len(cands), 3)

    def test_hold_with_tape_attached_is_kept(self):
        # Phone-resolution frame, where tape may be up to 2.5% of the width
        # (54px). A 44px chip with a 20px strip hanging off it used to be
        # deleted as tape because the union is thin and elongated; now the
        # strip is trimmed off and the chip kept with a tight box.
        img = np.full((1600, 2160, 3), (60, 60, 60), dtype=np.uint8)
        cv2.circle(img, (1000, 600), 22, BLUE, -1)
        cv2.rectangle(img, (990, 620), (1010, 820), BLUE, -1)
        cv2.imwrite(self.path, img)
        cands = find_hold_candidates(self.path, "Blue")
        self.assertEqual(len(cands), 1)
        self.assertTrue(_covers(cands, 1000, 600))
        self.assertLess(cands[0]["by"] + cands[0]["bh"], 700, "the tape strip should be trimmed off the box")

    def test_lone_tape_strip_is_dropped(self):
        img = np.full((1600, 2160, 3), (60, 60, 60), dtype=np.uint8)
        cv2.rectangle(img, (990, 600), (1010, 820), BLUE, -1)       # a strip on its own
        cv2.circle(img, (500, 400), 40, BLUE, -1)                    # a plain hold, as control
        cv2.imwrite(self.path, img)
        cands = find_hold_candidates(self.path, "Blue")
        self.assertEqual(len(cands), 1)
        self.assertTrue(_covers(cands, 500, 400))

    def test_black_ceiling_follows_exposure(self):
        # On a brightly lit wall black holds photograph at V~55-60, above the
        # old fixed ceiling of 50.
        img = np.full((1000, 800, 3), (210, 210, 210), dtype=np.uint8)
        for x, y in ((200, 300), (500, 600)):
            cv2.circle(img, (x, y), 30, (58, 58, 58), -1)
        cv2.imwrite(self.path, img)
        cands = find_hold_candidates(self.path, "Black")
        self.assertEqual(len(cands), 2)


if __name__ == "__main__":
    unittest.main()


class ValidationEvidenceTests(unittest.TestCase):
    """The validation prompt must carry the colour stage's evidence: which
    markers are highest/lowest and which came out of one blob split at a
    pinch, so a finish hold next to tape or a stacked pair is not thrown away."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "wall.jpg")

    def tearDown(self):
        self.tmp.cleanup()

    def _prompt(self, blobs):
        _wall(self.path, blobs=blobs)
        client = _FakeClient(holds=[{"number": 1, "hold_type": "jug", "best_use": "handhold",
                                     "orientation": "top", "same_hold_as": None}])
        detect_and_validate_holds(client, self.path, "Blue")
        return client.calls[0]["messages"][0]["content"][-1]["text"]

    def test_highest_and_lowest_markers_are_flagged(self):
        text = self._prompt([(200, 800, 30), (400, 500, 25), (600, 200, 35)])
        self.assertIn("HIGHEST marker", text)
        self.assertIn("LOWEST marker", text)
        self.assertIn("Never discard the highest or lowest candidate", text)
        self.assertIn("one tap for the climber", text)

    def test_split_pair_is_reported_as_shape_evidence(self):
        # two fat circles joined by a thin neck: one blob, split by the watershed
        img = np.full((1000, 800, 3), (60, 60, 60), dtype=np.uint8)
        cv2.circle(img, (300, 500), 70, (220, 90, 20), -1)
        cv2.circle(img, (520, 500), 70, (220, 90, 20), -1)
        cv2.rectangle(img, (300, 490), (520, 510), (220, 90, 20), -1)
        cv2.imwrite(self.path, img)
        cands = find_hold_candidates(self.path, "Blue")
        self.assertEqual(len(cands), 2)
        self.assertIsNotNone(cands[0]["split_group"])
        self.assertEqual(cands[0]["split_group"], cands[1]["split_group"])
        client = _FakeClient(holds=[])
        with self.assertRaises(DetectionError):
            detect_and_validate_holds(client, self.path, "Blue")
        text = client.calls[0]["messages"][0]["content"][-1]["text"]
        self.assertIn("SHAPE ANALYSIS NOTES", text)
        self.assertIn("Candidates 1 and 2 were ONE colour blob", text)

    def test_lone_holds_carry_no_split_group(self):
        _wall(self.path, blobs=[(200, 800, 30), (600, 200, 35)])
        for c in find_hold_candidates(self.path, "Blue"):
            self.assertIsNone(c["split_group"])
