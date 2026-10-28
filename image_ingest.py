"""Normalise an uploaded wall photo before anything else touches it.

Phone photos arrive as 8-12 MP JPEGs carrying an EXIF orientation tag.
OpenCV honours that tag when it reads a file and PIL does not, so without
one normalisation step the detector and the annotated image disagree about
which way is up and every marker lands in the wrong place on portrait
shots. Full-resolution photos also cost RAM on a 1 GB host and bandwidth on
every tap of the hold editor, and buy nothing in accuracy: Claude vision
downsizes anything above ~1.15 MP anyway and the detector's kernels are
already resolution-scaled.

prepare_upload() is the single entry point: it validates, applies the
orientation, strips EXIF, caps the longest side, and writes a plain RGB
JPEG. Everything downstream reads that file.
"""

import io

from PIL import Image, ImageOps

MAX_UPLOAD_BYTES  = 20 * 1024 * 1024   # bigger than any phone JPEG
MAX_SOURCE_PIXELS = 60_000_000         # refuse to decode above this (bomb guard)
MAX_SIDE          = 1600               # longest side after normalisation


class ImageRejected(ValueError):
    """The upload is not something we are willing to process. The message
    is written for the climber and safe to show in the UI."""


def prepare_upload(data, dest_path, max_side=MAX_SIDE):
    """Validate `data` (bytes of an uploaded image), normalise it, and write
    it to `dest_path` as a JPEG. Returns (width, height) of the written file.
    Raises ImageRejected with a user-facing message on bad input."""
    if not data:
        raise ImageRejected("The upload was empty — try taking the photo again.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageRejected(
            f"That photo is {len(data) / 1e6:.0f} MB — please use one under "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    try:
        img = Image.open(io.BytesIO(data))
        width, height = img.size          # header only — nothing decoded yet
    except Exception:
        raise ImageRejected("That file doesn't look like a photo — please upload a JPEG or PNG.")

    if width * height > MAX_SOURCE_PIXELS:
        raise ImageRejected(
            f"That photo is {width}×{height} — far larger than any phone photo. "
            "Please use a smaller one."
        )

    try:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except Exception:
        raise ImageRejected("That photo couldn't be decoded — try re-saving it or taking it again.")

    # Only ever shrink; thumbnail keeps the aspect ratio.
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    img.save(dest_path, format="JPEG", quality=90, optimize=True)   # no EXIF written
    return img.size
