"""prepare_upload must make OpenCV and PIL agree on orientation, shrink big
photos, and reject junk with a user-facing message — no API calls."""

import io
import os
import tempfile
import unittest

import cv2
from PIL import Image

import image_ingest
from image_ingest import ImageRejected, prepare_upload


def _jpeg_bytes(size, orientation=None, colour=(120, 120, 120)):
    img = Image.new("RGB", size, colour)
    buf = io.BytesIO()
    kwargs = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kwargs["exif"] = exif.tobytes()
    img.save(buf, format="JPEG", **kwargs)
    return buf.getvalue()


class PrepareUploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = os.path.join(self.tmp.name, "wall.jpg")

    def tearDown(self):
        self.tmp.cleanup()

    def test_exif_rotated_portrait_photo_comes_out_portrait_for_both_libraries(self):
        # A phone stores a portrait shot as 300x200 landscape pixels + orientation 6
        data = _jpeg_bytes((300, 200), orientation=6)
        w, h = prepare_upload(data, self.dest)
        self.assertEqual((w, h), (200, 300))
        self.assertEqual(Image.open(self.dest).size, (200, 300))
        rows, cols = cv2.imread(self.dest).shape[:2]
        self.assertEqual((cols, rows), (200, 300))
        # and the EXIF tag is gone, so nothing can re-rotate it later
        self.assertFalse(Image.open(self.dest).getexif().get(0x0112))

    def test_large_photo_is_capped_to_max_side_keeping_aspect(self):
        data = _jpeg_bytes((4000, 3000))
        self.assertEqual(prepare_upload(data, self.dest), (1600, 1200))

    def test_small_photo_is_not_upscaled(self):
        data = _jpeg_bytes((640, 480))
        self.assertEqual(prepare_upload(data, self.dest), (640, 480))

    def test_png_input_becomes_rgb_jpeg(self):
        img = Image.new("RGBA", (300, 300), (255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        prepare_upload(buf.getvalue(), self.dest)
        out = Image.open(self.dest)
        self.assertEqual(out.format, "JPEG")
        self.assertEqual(out.mode, "RGB")

    def test_rejects_empty_oversized_and_non_image(self):
        with self.assertRaises(ImageRejected):
            prepare_upload(b"", self.dest)
        with self.assertRaises(ImageRejected):
            prepare_upload(b"x" * (image_ingest.MAX_UPLOAD_BYTES + 1), self.dest)
        with self.assertRaises(ImageRejected):
            prepare_upload(b"definitely not a jpeg", self.dest)
        self.assertFalse(os.path.exists(self.dest))

    def test_rejects_absurd_pixel_counts_before_decoding(self):
        original = image_ingest.MAX_SOURCE_PIXELS
        image_ingest.MAX_SOURCE_PIXELS = 1000
        try:
            with self.assertRaises(ImageRejected) as ctx:
                prepare_upload(_jpeg_bytes((100, 100)), self.dest)
            self.assertIn("100×100", str(ctx.exception))
        finally:
            image_ingest.MAX_SOURCE_PIXELS = original


if __name__ == "__main__":
    unittest.main()
