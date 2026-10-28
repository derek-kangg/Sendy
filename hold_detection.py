"""OpenCV hold-candidate detection.

Finds colour-matched hold candidates with pixel-accurate coordinates.
Includes shape- and colour-level corrections learned from real gym photos:
- watershed splitting for merged blobs (two touching holds detected as one)
- a tape trimmer (route tape is route-coloured but thin: a lone strip is
  dropped, a strip stuck to a hold is shaved off and the hold kept)
- wall-paint rejection (a white route on a white wall must not segment the
  wall itself) with a local-contrast search for holds on that paint
- an exposure-adaptive black ceiling (black holds in a bright gym sit well
  above a fixed V=50; a dim photo would turn the whole wall black)
Claude's vision pass (in app.py) then labels type/use/orientation.
"""

import base64
import io
import logging

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VALIDATION_MODEL  = "claude-sonnet-4-6"
REQUEST_TIMEOUT_S = 90.0   # one vision call; the SDK default is 10 minutes
MAX_CANDIDATES    = 60     # past this the photo needs cropping, not labelling

log = logging.getLogger("sendy.detection")


class DetectionError(Exception):
    """Hold detection could not complete. The message is written for the climber."""


class NoHoldsFound(DetectionError):
    """The colour mask matched nothing worth calling a hold."""

    def __init__(self, colour):
        self.colour = colour
        super().__init__(
            f"No {colour.lower()} holds found in this photo. Check the route colour, "
            "move closer so the holds are bigger, or add them by hand below."
        )

# HSV ranges per route colour: a list of (low, high) bounds whose masks are
# OR-ed together, so a colour that straddles the hue wrap (red) is one entry
# like any other. OpenCV hue runs 0-180.
COLOUR_RANGES = {
    "Black":  [((0, 0, 0),       (180, 90, 80))],     # V ceiling tightened per photo, see _colour_mask
    "Blue":   [((90, 50, 50),    (130, 255, 255))],
    "Red":    [((0, 100, 100),   (10, 255, 255)),     # red straddles the hue wrap ...
               ((170, 100, 100), (180, 255, 255))],   # ... crimson / deep reds sit at 170-180
    "Green":  [((40, 50, 50),    (80, 255, 255))],
    "Orange": [((10, 100, 100),  (25, 255, 255))],
    "Pink":   [((140, 40, 100),  (180, 255, 255))],   # hot pink is hue 165-178; chalked pink drops to S~45
    "White":  [((0, 0, 170),     (180, 35, 255))],    # near-white only: cream on painted walls is found by _holds_on_painted_wall
    "Yellow": [((25, 100, 100),  (35, 255, 255))],
    "Purple": [((130, 50, 50),   (160, 255, 255))],
}

# Black: pixels this dark are black whatever their saturation (S is noise
# next to V=0), and the ceiling for the rest follows the photo's exposure.
BLACK_DEEP_V     = 35
BLACK_V_FRACTION = 0.4        # of the photo's median V ...
BLACK_V_LIMITS   = (50, 80)   # ... clamped to this

# One blob this big that also spans a good part of the frame is the painted
# wall (or the mats), not a hold.
WALL_MIN_FRACTION = 0.06      # of the image area ...
WALL_MIN_SPAN     = 0.4       # ... and of the image width or height
WALL_FULL_SPAN    = 0.7       # this wide or tall is never a hold, whatever its area

# White holds on a white wall: a pixel counts as a hold when its Lab
# deviation from the surrounding paint - brighter, tinted either way in a,
# yellower in b - reaches this many (robust) standard deviations combined,
# and it is still a whitish pixel.
PAINT_DEVIATION_Z = 3.5
PAINT_WHITE_RANGE = ((0, 0, 150), (180, 70, 255))

TAPE_ASPECT = 2.3             # longer than this, relative to its width, is a strip
SLIVER_ASPECT = 4.0           # a watershed piece this much longer than its real thickness is a boundary sliver
TAPE_CORE   = 0.6             # a hold outlives an opening at this fraction of its own thickest point; tape does not
TAPE_HOLD_THICKNESS = 0.4     # what outlives it must be this much of the tape width thick, and ...
TAPE_HOLD_FRACTION  = 0.1     # ... this much of the blob's area, or it is a lump on a streak


def _colour_mask(hsv, colour):
    """Union of the colour's HSV ranges. Black's V ceiling adapts to the
    photo: 0.4 x median brightness, clamped, plus everything near-black."""
    ranges = COLOUR_RANGES[colour]
    if colour == "Black":
        v_max = int(np.clip(BLACK_V_FRACTION * float(np.median(hsv[:, :, 2])), *BLACK_V_LIMITS))
        (low, high), = ranges
        ranges = [(low, (high[0], high[1], min(high[2], v_max))),
                  ((0, 0, 0), (180, 255, BLACK_DEEP_V))]
    mask = None
    for low, high in ranges:
        part = cv2.inRange(hsv, np.array(low), np.array(high))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    return mask


def _wall_paint(mask, img_w, img_h):
    """Mask of connected regions too big to be holds: the painted wall itself,
    or the mats along the bottom of the frame."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    wall = np.zeros_like(mask)
    min_area = WALL_MIN_FRACTION * img_w * img_h
    for i in range(1, n):
        _, _, w, h, area = stats[i]
        big = area >= min_area and (w >= WALL_MIN_SPAN * img_w or h >= WALL_MIN_SPAN * img_h)
        if big or w >= WALL_FULL_SPAN * img_w or h >= WALL_FULL_SPAN * img_h:
            wall[labels == i] = 255
    return wall


def _fill_holes(mask, img_w, img_h):
    """Fill enclosed gaps (holds of any colour sitting on the wall) so the
    wall region is one solid sheet; gaps touching the frame edge stay open."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(mask), connectivity=4)
    filled = mask.copy()
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if x > 0 and y > 0 and x + w < img_w and y + h < img_h:
            filled[labels == i] = 255
    return filled


def _holds_on_painted_wall(bgr, wall, img_w, img_h):
    """Holds of the wall's own colour, sitting on that wall. Inside the
    wall-paint region they show up as patches that depart from the local
    paint in Lab: brighter (glossy plastic on matte paint), tinted either way
    in a, yellower in b (cream on white). Chalk is brighter but bluer, and
    shadows and bolt holes are darker, so neither scores. The paint's own
    noise sets the scale, so wood grain or mottled paint do not light up.
    Returns a mask."""
    # Search the wall with its gaps filled (holds of any colour sit in those
    # gaps), but estimate the local paint colour from paint pixels only, or
    # a big orange hold would tint the background next to it and the paint
    # around it would read as a deviation.
    inside = _fill_holes(wall, img_w, img_h) > 0
    paint = (wall > 0).astype(np.float32)
    k = max(31, img_w // 6) | 1                       # local background window
    weight = np.maximum(cv2.blur(paint, (k, k)), 1e-3)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    score = np.zeros(wall.shape, dtype=np.float32)
    for ch, sigma_floor, signed in ((0, 3.0, True), (1, 1.0, False), (2, 1.0, True)):
        values = lab[:, :, ch]
        background = cv2.blur(values * paint, (k, k)) / weight
        resid = values - background
        r = resid[wall > 0]                           # paint pixels only set the scale
        sigma = max(sigma_floor, 1.4826 * float(np.median(np.abs(r - np.median(r)))))
        z = (np.maximum(resid, 0) if signed else np.abs(resid)) / sigma
        score += z * z
    whitish = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
                          np.array(PAINT_WHITE_RANGE[0]), np.array(PAINT_WHITE_RANGE[1]))
    found = inside & (score >= PAINT_DEVIATION_Z ** 2) & (whitish > 0)
    return np.uint8(found) * 255


def _strip_tape(contour, img_width, split_piece=False):
    """Route tape is a thin strip of the route colour. A lone strip is
    dropped; a strip attached to a hold is trimmed off and the hold kept.
    Returns a list of contours.

    Anything compact, or wider than tape ever is, passes untouched. A narrow
    elongated blob is either a lone strip or a small hold with a strip stuck
    to it: opening it with a disc scaled to its own thickest point tells
    them apart, because a strip is the same width everywhere and survives
    whole, while a hold outlives its tape as a compact piece. A piece must
    still be hold-sized: wood grain and chalk streaks leave compact lumps a
    few pixels thick, and those are not holds.

    A watershed piece (split_piece=True) is also dropped when it is long for
    its real thickness: the thin crescents a split leaves along a boundary.
    The ratio is generous, since a triangular volume is three times longer
    than its inscribed circle and a sliver five or more. Whole contours are
    spared that test, because a chalked-over hold reads as a thin ring with
    a compact bounding box and is still a hold."""
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    short, long_ = min(rw, rh), max(rw, rh)
    if short <= 0:
        return []
    max_tape_width = max(16, img_width * 0.025)
    elongated = long_ / short > TAPE_ASPECT

    x, y, w, h = cv2.boundingRect(contour)
    roi = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.drawContours(roi, [contour - [x - 1, y - 1]], -1, 255, thickness=cv2.FILLED)
    dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    thickness = 2.0 * float(dist.max())             # widest inscribed circle

    if split_piece and long_ > SLIVER_ASPECT * thickness:
        return []
    if not elongated or short >= max_tape_width:
        return [contour]

    radius = max(1, int(TAPE_CORE * dist.max()))
    body = np.uint8(dist > radius) * 255
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    opened = cv2.bitwise_and(cv2.dilate(body, disc), roi)
    pieces, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                 offset=(x - 1, y - 1))
    kept = []
    blob_area = max(cv2.contourArea(contour), 1.0)
    for piece in pieces:
        (_, _), (pw, ph), _ = cv2.minAreaRect(piece)
        if min(pw, ph) <= 0 or max(pw, ph) / min(pw, ph) > TAPE_ASPECT:
            continue
        piece_mask = np.zeros_like(roi)
        cv2.drawContours(piece_mask, [piece - [x - 1, y - 1]], -1, 255, thickness=cv2.FILLED)
        piece_thickness = 2.0 * float(dist[piece_mask > 0].max())
        if (piece_thickness >= TAPE_HOLD_THICKNESS * max_tape_width
                and cv2.contourArea(piece) >= TAPE_HOLD_FRACTION * blob_area):
            kept.append(piece)
    return kept


def _is_tape(contour, img_width):
    """True when nothing hold-like is left once tape is trimmed."""
    return not _strip_tape(contour, img_width)


def _split_merged_blob(mask, contour):
    """Try to split one contour that may contain 2+ touching holds.

    Watershed on distance-transform cores, then PAIRWISE neck analysis:
    two adjacent pieces re-merge when the neck between them is nearly as
    thick as the smaller piece's core (one chalky/patchy hold), and stay
    apart when they meet at a thin pinch (two real holds). Junk cores get
    absorbed by their neighbours instead of poisoning a global test.

    Returns a list of contours (full-image coordinates) if a genuine split
    remains, else None."""
    x, y, w, h = cv2.boundingRect(contour)

    # Isolated ROI mask of just this contour (not neighbours)
    roi = np.zeros((h, w), dtype=np.uint8)
    shifted = contour - [x, y]
    cv2.drawContours(roi, [shifted], -1, 255, thickness=cv2.FILLED)

    dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    peak = dist.max()
    if peak <= 0:
        return None

    core_thresh = max(0.35 * peak, 5.0)
    _, sure_fg = cv2.threshold(dist, core_thresh, 255, 0)
    sure_fg = np.uint8(sure_fg)
    n_labels, markers = cv2.connectedComponents(sure_fg)
    if n_labels <= 2:  # background + one core -> genuinely one hold
        return None

    # Watershed to assign the ambiguous pixels between the cores
    markers = markers + 1
    unknown = cv2.subtract(roi, sure_fg)
    markers[unknown == 255] = 0
    cv2.watershed(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), markers)

    labels = [lb for lb in range(2, n_labels + 1) if np.any(markers == lb)]
    if len(labels) < 2:
        return None

    piece_masks = {lb: np.uint8(markers == lb) for lb in labels}
    peaks = {lb: float(dist[markers == lb].max()) for lb in labels}
    kern = np.ones((5, 5), np.uint8)
    dilated = {lb: cv2.dilate(piece_masks[lb], kern) for lb in labels}

    # union-find over pieces; merge pairs whose shared neck is fat
    parent = {lb: lb for lb in labels}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            shared = (dilated[a] > 0) & (dilated[b] > 0) & (roi > 0)
            if not shared.any():
                continue
            neck = float(dist[shared].max())
            if neck >= 0.65 * min(peaks[a], peaks[b]):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

    groups = {}
    for lb in labels:
        groups.setdefault(find(lb), []).append(lb)
    if len(groups) < 2:
        return None  # everything merged back -> one hold

    blob_area = cv2.contourArea(contour)
    pieces = []
    for members in groups.values():
        gmask = np.zeros_like(roi)
        for lb in members:
            gmask = cv2.bitwise_or(gmask, piece_masks[lb] * 255)
        cs, _ = cv2.findContours(gmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                 offset=(x, y))
        if not cs:
            continue
        biggest = max(cs, key=cv2.contourArea)
        # keep only substantial pieces — slivers mean a bad split
        if cv2.contourArea(biggest) > 0.025 * blob_area:
            pieces.append(biggest)

    return pieces if len(pieces) >= 2 else None


def find_hold_candidates(image_path, colour, min_area=80):
    """Detect hold candidates for a route colour. Returns a list of dicts:
    {bx, by, bw, bh, x, y, size} in full-image pixel coordinates."""
    cv_image = cv2.imread(image_path)
    if cv_image is None:
        raise DetectionError("Couldn't read that photo — please upload it again.")
    img_h, img_w = cv_image.shape[:2]
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    mask = _colour_mask(hsv, colour)

    # CLOSE kernel scales with resolution so chalk patches on a hold do not
    # fragment it into multiple detections (phone photos are ~2000px+)
    k = max(5, img_w // 200) | 1
    close_kernel = np.ones((k, k), np.uint8)
    open_kernel = np.ones((5, 5), np.uint8)

    def _tidy(m):
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_kernel)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, open_kernel)

    mask = _tidy(mask)

    # A blob the size of the wall is the wall. Drop it, and for white routes
    # on white paint look inside it for holds that differ from the paint.
    wall = _wall_paint(mask, img_w, img_h)
    if wall.any():
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(wall))
        if colour == "White":
            recovered = _tidy(_holds_on_painted_wall(cv_image, wall, img_w, img_h))
            recovered = cv2.bitwise_and(recovered, cv2.bitwise_not(_wall_paint(recovered, img_w, img_h)))
            mask = cv2.bitwise_or(mask, recovered)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Split any large blob that actually contains multiple touching holds.
    # Scale the "large" threshold with resolution (phone photos are ~2000px+).
    split_threshold = max(1500, (img_w / 700.0) ** 2 * 350)
    processed = []  # (contour, split_group) — pieces of one split share a group id
    split_group = 0
    for contour in contours:
        if cv2.contourArea(contour) >= split_threshold:
            pieces = _split_merged_blob(mask, contour)
            if pieces:
                split_group += 1
                processed.extend((p, split_group) for p in pieces)
                continue
        processed.append((contour, None))

    candidates = []
    scale2 = (img_w / 770.0) ** 2   # size classes relative to resolution (tuned at ~770px wide)
    for contour, grp in processed:
        if cv2.contourArea(contour) < min_area:
            continue
        for piece in _strip_tape(contour, img_w, split_piece=grp is not None):
            area = cv2.contourArea(piece)
            if area < min_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(piece)
            if bw < 12 or bh < 12:
                continue
            if by < img_h * 0.03:
                continue

            cx, cy = bx + bw // 2, by + bh // 2
            if area < 500 * scale2:
                size = "small"
            elif area < 2000 * scale2:
                size = "medium"
            else:
                size = "large"

            candidates.append({"bx": bx, "by": by, "bw": bw, "bh": bh,
                               "x": cx, "y": cy, "size": size, "grp": grp})

    deduped = _dedup_overlaps(candidates)
    for c in deduped:
        c.pop("grp", None)
    return deduped


def marker_metrics(img_width):
    """Marker radius and font scaled to image resolution — phone photos are
    ~2000px+ and fixed-size markers become unreadably small."""
    r = max(15, img_width // 70)
    try:
        font = ImageFont.load_default(size=max(11, int(r * 1.1)))
    except TypeError:
        font = ImageFont.load_default()
    return r, font


_SIZE_LABELS = {
    "small":  "small (likely foothold)",
    "medium": "medium (likely handhold)",
    "large":  "large (likely handhold)",
}


def _resolve_merge_groups(validated, n):
    """Group candidate numbers by Claude's same_hold_as links (with cycle
    and range guards). Returns {root_number: [member_numbers]}."""
    def root_of(i):
        seen = {i}
        while True:
            t = validated.get(i, {}).get("same_hold_as")
            if not isinstance(t, int) or t == i or t < 1 or t > n or t in seen:
                return i
            seen.add(t)
            i = t

    groups = {}
    for i in range(1, n + 1):
        groups.setdefault(root_of(i), []).append(i)
    return groups


def detect_and_validate_holds(client, image_path, colour):
    """
    Two-pass hold detection:
    Pass 1 — OpenCV finds candidates by colour (pixel-accurate coordinates;
              oversplitting is acceptable here).
    Pass 2 — Claude sees the numbered image and (a) labels each hold's type,
              use, and orientation, (b) merges markers that sit on the same
              physical hold, (c) flags tape/non-hold markers for removal.
    Returns: (annotated PIL image, holds list)
    """
    candidates = find_hold_candidates(image_path, colour)

    if not candidates:
        raise NoHoldsFound(colour)
    if len(candidates) > MAX_CANDIDATES:
        # Keep the biggest blobs; the rest are almost always noise, and past
        # this count the validation call itself becomes unreliable.
        log.warning("%d %s candidates — keeping the %d largest", len(candidates), colour, MAX_CANDIDATES)
        candidates = sorted(candidates, key=lambda c: c["bw"] * c["bh"], reverse=True)[:MAX_CANDIDATES]
        candidates.sort(key=lambda c: (c["y"], c["x"]))

    # Draw numbered candidates on a preview image for Claude to assess
    preview = Image.open(image_path).convert("RGB")
    preview_draw = ImageDraw.Draw(preview)
    mr, mfont = marker_metrics(preview.width)
    for i, c in enumerate(candidates, 1):
        cx, cy = c["x"], c["y"]
        preview_draw.ellipse([cx - mr, cy - mr, cx + mr, cy + mr], fill="#fb923c")
        preview_draw.text((cx, cy), str(i), fill="black", font=mfont, anchor="mm")

    buf = io.BytesIO()
    preview.save(buf, format="JPEG")
    image_data = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Also send the clean photo — markers cover exactly the details needed
    # to judge whether neighbouring markers share one hold
    raw = Image.open(image_path).convert("RGB")
    raw_buf = io.BytesIO()
    raw.save(raw_buf, format="JPEG")
    raw_data = base64.b64encode(raw_buf.getvalue()).decode("utf-8")

    candidate_list = "\n".join(
        f"Candidate {i}: x={c['x']}, y={c['y']}, size={c['size']}"
        for i, c in enumerate(candidates, 1)
    )

    tool = {
        "name": "submit_validated_holds",
        "description": "Submit validation results for each detected hold candidate",
        "input_schema": {
            "type": "object",
            "properties": {
                "holds": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "number":       {"type": "integer"},
                            "hold_type":    {"type": "string", "enum": ["jug", "crimp", "sloper", "pinch", "pocket", "edge", "volume", "chip", "tape", "unknown"]},
                            "best_use":     {"type": "string", "enum": ["handhold", "foothold", "both"]},
                            "orientation":  {"type": "string", "enum": ["top", "undercling", "side-pull left", "side-pull right", "unknown"], "description": "Which direction the usable surface faces: top = pull down on it (normal), undercling = usable surface faces down so palm goes up, side-pull left/right = vertical edge pulled sideways"},
                            "same_hold_as": {"type": ["integer", "null"], "description": "If this marker sits on the SAME physical hold as a LOWER-numbered marker (fragments of one hold detected separately), the lower number. Otherwise null."}
                        },
                        "required": ["number", "hold_type", "best_use", "orientation", "same_hold_as"]
                    }
                }
            },
            "required": ["holds"]
        }
    }

    # ~45 output tokens per candidate; a fixed 1500 silently truncated busy colours
    max_tokens = min(8000, 300 + 70 * len(candidates))
    response = client.messages.create(
        model=VALIDATION_MODEL,
        max_tokens=max_tokens,
        timeout=REQUEST_TIMEOUT_S,
        tools=[tool],
        tool_choice={"type": "any"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": raw_data}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                {"type": "text", "text": f"""Image 1 is a clean photo of a climbing wall. Image 2 is the same photo with {len(candidates)} numbered candidate markers detected by colour analysis (target colour: {colour}). Use image 1 to see details the markers cover in image 2. Automated detection makes two kinds of mistakes you must correct:
- One physical hold can carry MULTIPLE markers (chalk patches or two-tone colouring split it). Look carefully at each cluster of nearby markers and decide whether they sit on one hold or several.
- A marker can sit on route TAPE or another non-hold object instead of a hold.

{candidate_list}

For every candidate number, report:
1. hold_type: jug / crimp / sloper / pinch / pocket / edge / volume / chip — or "tape" if the marker is on tape or not on any hold
2. best_use: handhold, foothold, or both (small chips are typically footholds)
3. orientation — which way the usable surface faces: top (pull down, normal), undercling (surface faces DOWN, palm-up), side-pull left or right. Look at shadows and shape.
4. same_hold_as: if this marker is on the SAME physical hold as a lower-numbered marker, give that lower number; otherwise null.

Assess every single candidate — do not skip any."""}
            ]
        }]
    )

    validated = {}
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_validated_holds":
            for h in (block.input or {}).get("holds") or []:
                if isinstance(h, dict) and isinstance(h.get("number"), int):
                    validated[h["number"]] = h
            break

    if getattr(response, "stop_reason", None) == "max_tokens":
        log.warning("validation output truncated at %d tokens for %d candidates (%d labelled)",
                    max_tokens, len(candidates), len(validated))
        if not validated:
            raise DetectionError(
                "The hold check was cut short — try cropping the photo to just this route and detecting again."
            )
    elif not validated:
        raise DetectionError("The hold check returned nothing usable — please try again.")

    groups = _resolve_merge_groups(validated, len(candidates))

    # Build final holds list and annotated image
    final_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(final_image)
    mr, mfont = marker_metrics(final_image.width)
    box_w = max(3, final_image.width // 500)
    final_holds = []
    new_number = 1

    for root in sorted(groups):
        v = validated.get(root, {"hold_type": "unknown", "best_use": "both"})
        if v.get("hold_type") == "tape":
            continue

        members = [candidates[m - 1] for m in groups[root]]
        bx = min(m["bx"] for m in members)
        by = min(m["by"] for m in members)
        bw = max(m["bx"] + m["bw"] for m in members) - bx
        bh = max(m["by"] + m["bh"] for m in members) - by
        cx, cy = bx + bw // 2, by + bh // 2
        size = max((m["size"] for m in members), key=lambda s: _SIZE_RANK[s])

        best_use  = v.get("best_use", "both")
        hold_type = v.get("hold_type", "unknown")

        if best_use == "foothold" or size == "small":
            size_label = "small (likely foothold)"
        else:
            size_label = _SIZE_LABELS[size]

        final_holds.append({
            "number":      new_number,
            "x":           cx,
            "y":           cy,
            "size":        size_label,
            "hold_type":   hold_type,
            "best_use":    best_use,
            "orientation": v.get("orientation", "unknown"),
            "bx": bx, "by": by, "bw": bw, "bh": bh,
        })

        draw.rectangle([bx, by, bx + bw, by + bh], outline="#fb923c", width=box_w)
        draw.ellipse([cx - mr, cy - mr, cx + mr, cy + mr], fill="#fb923c")
        draw.text((cx, cy), str(new_number), fill="black", font=mfont, anchor="mm")
        new_number += 1

    return final_image, final_holds


def redraw_annotation(image_path, holds):
    """Redraw the numbered annotation from a holds list (after manual
    add/remove edits). Boxes are drawn where bbox info exists."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    mr, mfont = marker_metrics(image.width)
    box_w = max(3, image.width // 500)

    for h in holds:
        if all(k in h for k in ("bx", "by", "bw", "bh")):
            draw.rectangle([h["bx"], h["by"], h["bx"] + h["bw"], h["by"] + h["bh"]],
                           outline="#fb923c", width=box_w)
        draw.ellipse([h["x"] - mr, h["y"] - mr, h["x"] + mr, h["y"] + mr], fill="#fb923c")
        draw.text((h["x"], h["y"]), str(h["number"]), fill="black", font=mfont, anchor="mm")

    return image


_SIZE_RANK = {"small": 0, "medium": 1, "large": 2}


def _dedup_overlaps(candidates):
    """Merge candidates whose boxes substantially overlap (>=40% of the
    smaller box) — usually chalk-fragmented pieces of one hold."""
    merged = True
    while merged:
        merged = False
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                # never re-merge pieces the watershed deliberately split apart
                if a.get("grp") is not None and a.get("grp") == b.get("grp"):
                    continue
                ix = max(0, min(a["bx"] + a["bw"], b["bx"] + b["bw"]) - max(a["bx"], b["bx"]))
                iy = max(0, min(a["by"] + a["bh"], b["by"] + b["bh"]) - max(a["by"], b["by"]))
                inter = ix * iy
                smaller = min(a["bw"] * a["bh"], b["bw"] * b["bh"])
                if smaller > 0 and inter / smaller >= 0.4:
                    bx = min(a["bx"], b["bx"])
                    by = min(a["by"], b["by"])
                    bw = max(a["bx"] + a["bw"], b["bx"] + b["bw"]) - bx
                    bh = max(a["by"] + a["bh"], b["by"] + b["bh"]) - by
                    candidates[i] = {
                        "bx": bx, "by": by, "bw": bw, "bh": bh,
                        "x": bx + bw // 2, "y": by + bh // 2,
                        "size": max(a["size"], b["size"], key=lambda s: _SIZE_RANK[s]),
                    }
                    del candidates[j]
                    merged = True
                    break
            if merged:
                break
    return candidates
