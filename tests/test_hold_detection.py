"""Detection failure modes must surface as typed, user-facing errors instead
of empty results or crashes. Uses synthetic images and a fake client — no
API calls."""

import os
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

import hold_detection
from hold_detection import (DetectionError, NoHoldsFound, detect_and_validate_holds,
                            find_hold_candidates)


def _wall(path, blobs=(), size=(800, 1000), bgr=(60, 60, 60)):
    """Grey wall with filled blue circles at the given (x, y, r) positions."""
    img = np.full((size[1], size[0], 3), bgr, dtype=np.uint8)
    for x, y, r in blobs:
        cv2.circle(img, (x, y), r, (220, 90, 20), -1)     # BGR blue
    cv2.imwrite(path, img)
    return path


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


if __name__ == "__main__":
    unittest.main()
