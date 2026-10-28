"""OpenCV hold-candidate detection.

Finds colour-matched hold candidates with pixel-accurate coordinates.
Includes two shape-level corrections learned from real gym photos:
- watershed splitting for merged blobs (two touching holds detected as one)
- a tape filter (route tape is route-coloured but thin and elongated)
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

log = logging.getLogger("climbai.detection")


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

COLOUR_RANGES = {
    "Black":  [(0, 0, 0),      (180, 80, 50)],
    "Blue":   [(90, 50, 50),   (130, 255, 255)],
    "Red":    [(0, 100, 100),  (10, 255, 255)],
    "Green":  [(40, 50, 50),   (80, 255, 255)],
    "Orange": [(10, 100, 100), (25, 255, 255)],
    "Pink":   [(140, 50, 100), (170, 255, 255)],
    "White":  [(0, 0, 180),    (180, 30, 255)],
    "Yellow": [(25, 100, 100), (35, 255, 255)],
    "Purple": [(130, 50, 50),  (160, 255, 255)],
}


def _is_tape(contour, img_width):
    """Route tape: thin, elongated strip of the route colour."""
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    short, long_ = min(rw, rh), max(rw, rh)
    if short <= 0:
        return True
    max_tape_width = max(16, img_width * 0.025)
    return (long_ / short) > 2.3 and short < max_tape_width


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

    low, high = COLOUR_RANGES[colour]
    mask = cv2.inRange(hsv, np.array(low), np.array(high))

    # CLOSE kernel scales with resolution so chalk patches on a hold do not
    # fragment it into multiple detections (phone photos are ~2000px+)
    k = max(5, img_w // 200) | 1
    close_kernel = np.ones((k, k), np.uint8)
    open_kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

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
    for contour, grp in processed:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw < 12 or bh < 12:
            continue
        if by < img_h * 0.03:
            continue
        if _is_tape(contour, img_w):
            continue

        cx, cy = bx + bw // 2, by + bh // 2

        # Size classes relative to resolution (original tuning was ~770px wide)
        scale2 = (img_w / 770.0) ** 2
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
