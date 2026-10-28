"""Render the suggested beta as an animated stick-figure GIF over the wall photo.

The figure is driven by the body-state timeline (LH/RH/LF/RF -> hold) that the
sequence generator tracks. Between two states, only the limbs that changed
animate — travelling in a slight arc — while the torso is re-derived every
frame from the four contact points, so the body follows naturally.
"""

import io
import math
from PIL import Image, ImageDraw, ImageFont

ACCENT  = "#fb923c"
OUTLINE = "#0f1011"
BODY    = "#ececec"
JOINTS  = ("LH", "RH", "LF", "RF")

LIMB_COLOURS = {
    "LH": "#facc15",  # left hand  — yellow
    "RH": "#fb923c",  # right hand — orange
    "LF": "#4ade80",  # left foot  — green
    "RF": "#38bdf8",  # right foot — blue
}
LIMB_NAMES = {"LH": "L hand", "RH": "R hand", "LF": "L foot", "RF": "R foot"}


def _limb_joint(limb):
    """Map a sequence limb string to its joint key (None for both/unknown)."""
    if "left" in limb:
        return "LH" if "hand" in limb else "LF"
    if "right" in limb:
        return "RH" if "hand" in limb else "RF"
    return None

FRAMES_PER_MOVE = 7   # animation frames per move
HOLD_FRAMES     = 4   # frames the pose is held after each move
INTRO_FRAMES    = 6   # frames showing the start position


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _ease(t):
    return t * t * (3 - 2 * t)  # smoothstep


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mid(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def compute_keypose(state, hold_px, dims):
    """Concrete pixel positions for all four joints in a given body state.
    Feet that are flagging/smearing (hold None) hang relative to the body."""
    leg = dims["seg_leg"] * 2
    pts = {}
    for j in JOINTS:
        h = state.get(j)
        if h is not None and h in hold_px:
            pts[j] = hold_px[h]

    hands      = [pts[j] for j in ("LH", "RH") if j in pts]
    feet_known = [pts[j] for j in ("LF", "RF") if j in pts]

    if hands:
        hands_mid = _mid(hands)
    elif feet_known:  # no hand on a hold (loaded/old data) — stand above the feet
        fm = _mid(feet_known)
        hands_mid = (fm[0], fm[1] - dims["torso"] - leg)
    else:             # degenerate — park mid-image
        hands_mid = (leg * 2, leg * 2)

    if feet_known:
        feet_mid = _mid(feet_known)
    else:
        feet_mid = (hands_mid[0], hands_mid[1] + dims["torso"] + leg)

    # A hand that is not on any hold hangs beside the other one rather than
    # crashing the renderer
    for j, side in (("LH", -1), ("RH", 1)):
        if j not in pts:
            pts[j] = (hands_mid[0] + side * dims["sh_hw"], hands_mid[1])

    hip = _lerp(hands_mid, feet_mid, 0.62)

    for j, side in (("LF", -1), ("RF", 1)):
        if j not in pts:
            pts[j] = (hip[0] + side * 0.45 * leg, hip[1] + 0.55 * leg)

    return pts


def _solve_joint(root, tip, seg_len, prefer_dir):
    """Two-bone IK: position of the elbow/knee for a limb with two equal
    segments of seg_len. Fully extended limbs come out straight; short
    reaches bend deeply. prefer_dir picks the anatomical bend side."""
    d = _dist(root, tip)
    mx, my = _mid([root, tip])

    if d >= 2 * seg_len or d < 1e-6:
        return (mx, my)  # straight (or degenerate)

    h = math.sqrt(max(seg_len * seg_len - (d / 2) * (d / 2), 0.0))
    dx, dy = (tip[0] - root[0]) / d, (tip[1] - root[1]) / d
    p1 = (-dy, dx)
    p2 = (dy, -dx)
    perp = p1 if (p1[0] * prefer_dir[0] + p1[1] * prefer_dir[1]) >= \
                 (p2[0] * prefer_dir[0] + p2[1] * prefer_dir[1]) else p2
    return (mx + perp[0] * h, my + perp[1] * h)


def _draw_line(draw, a, b, width, colour):
    draw.line([a, b], fill=OUTLINE, width=width + 3)
    draw.line([a, b], fill=colour, width=width)


def _draw_limb(draw, root, tip, seg_len, prefer_dir, width, colour):
    joint = _solve_joint(root, tip, seg_len, prefer_dir)
    _draw_line(draw, root, joint, width, colour)
    _draw_line(draw, joint, tip, width, colour)


def draw_figure(draw, pts, dims, line_w):
    hands_mid = _mid([pts["LH"], pts["RH"]])
    feet_mid  = _mid([pts["LF"], pts["RF"]])

    # body axis (feet -> hands) and a leftward perpendicular
    ax, ay = hands_mid[0] - feet_mid[0], hands_mid[1] - feet_mid[1]
    alen = math.hypot(ax, ay) or 1.0
    ax, ay = ax / alen, ay / alen
    px, py = -ay, ax
    if px > 0:  # make perp point screen-left
        px, py = -px, -py

    # FIXED-length torso anchored between hands and feet — the body no
    # longer scales with the span, so big extensions look like a human at
    # full stretch instead of a growing giant
    torso = dims["torso"]
    center = _lerp(hands_mid, feet_mid, 0.46)
    shoulder = (center[0] + ax * torso / 2, center[1] + ay * torso / 2)
    hip      = (center[0] - ax * torso / 2, center[1] - ay * torso / 2)

    sh_hw, hip_hw = dims["sh_hw"], dims["hip_hw"]
    shoulder_l = (shoulder[0] + px * sh_hw, shoulder[1] + py * sh_hw)
    shoulder_r = (shoulder[0] - px * sh_hw, shoulder[1] - py * sh_hw)
    hip_l = (hip[0] + px * hip_hw, hip[1] + py * hip_hw)
    hip_r = (hip[0] - px * hip_hw, hip[1] - py * hip_hw)

    head_r = dims["head_r"]
    head_c = (shoulder[0] + ax * head_r * 1.6,
              shoulder[1] + ay * head_r * 1.6)

    seg_arm = dims["seg_arm"]
    seg_leg = dims["seg_leg"]

    # torso: shoulder line, hip line, spine
    _draw_line(draw, shoulder_l, shoulder_r, line_w, BODY)
    _draw_line(draw, hip_l, hip_r, line_w, BODY)
    _draw_line(draw, shoulder, hip, line_w + 1, BODY)

    # limbs — elbows bend out-and-down, knees frog out-and-slightly-up
    _draw_limb(draw, shoulder_l, pts["LH"], seg_arm, (-1.0,  0.5), line_w, LIMB_COLOURS["LH"])
    _draw_limb(draw, shoulder_r, pts["RH"], seg_arm, ( 1.0,  0.5), line_w, LIMB_COLOURS["RH"])
    _draw_limb(draw, hip_l,      pts["LF"], seg_leg, (-1.0, -0.25), line_w, LIMB_COLOURS["LF"])
    _draw_limb(draw, hip_r,      pts["RF"], seg_leg, ( 1.0, -0.25), line_w, LIMB_COLOURS["RF"])

    # head
    draw.ellipse([head_c[0] - head_r - 2, head_c[1] - head_r - 2,
                  head_c[0] + head_r + 2, head_c[1] + head_r + 2], fill=OUTLINE)
    draw.ellipse([head_c[0] - head_r, head_c[1] - head_r,
                  head_c[0] + head_r, head_c[1] + head_r], fill=BODY)

    # hands/feet dots in their limb colour
    r = line_w + 3
    for j in JOINTS:
        x, y = pts[j]
        draw.ellipse([x - r, y - r, x + r, y + r], fill=OUTLINE)
        draw.ellipse([x - r + 1, y - r + 1, x + r - 1, y + r - 1],
                     fill=LIMB_COLOURS[j])


def _draw_legend(draw, font, img_w):
    """Small colour legend, top-right corner."""
    pad, row_h, sw = 5, 15, 9
    entries = [(LIMB_COLOURS[j], LIMB_NAMES[j]) for j in JOINTS]
    w = 74
    h = pad * 2 + row_h * len(entries)
    x0 = img_w - w - 8
    y0 = 8
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=OUTLINE)
    for i, (col, name) in enumerate(entries):
        y = y0 + pad + i * row_h
        draw.ellipse([x0 + pad, y + 2, x0 + pad + sw, y + 2 + sw], fill=col)
        draw.text((x0 + pad + sw + 5, y), name, fill="#ececec", font=font)


def _label_for(move):
    if move["move_number"] == 0:
        return "Start position"
    hold_str = f"hold {move['hold']}" if move["hold"] is not None else "wall"
    return f"Move {move['move_number']}: {move['limb']} -> {hold_str}"


def _draw_label(draw, text, font, img_w):
    pad = 6
    try:
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
    except Exception:
        tw, th = len(text) * 7, 12
    draw.rectangle([10, 10, 10 + tw + pad * 2, 10 + th + pad * 2], fill=OUTLINE)
    draw.text((10 + pad, 10 + pad), text, fill="#ececec", font=font)


def render_beta_gif(base_image, holds, sequence, states, height_cm=170,
                    out_width=480, frame_ms=100, wall_height_cm=400):
    """Build the animated GIF. Returns io.BytesIO with the GIF bytes."""
    img = base_image.convert("RGB")
    scale = out_width / img.width
    base = img.resize((out_width, max(1, int(img.height * scale))))

    hold_px = {h["number"]: (h["x"] * scale, h["y"] * scale) for h in holds}

    # Human proportions anchored to the climber's height, using the same
    # wall-scale assumption as the reachability graph
    px_cm = base.height / float(wall_height_cm or 400)
    dims = {
        "torso":   0.29  * height_cm * px_cm,
        "seg_arm": 0.21  * height_cm * px_cm,
        "seg_leg": 0.25  * height_cm * px_cm,
        "head_r":  0.055 * height_cm * px_cm,
        "sh_hw":   0.10  * height_cm * px_cm,
        "hip_hw":  0.065 * height_cm * px_cm,
    }
    line_w = max(3, out_width // 160)

    try:
        font = ImageFont.load_default(size=max(12, out_width // 40))
        legend_font = ImageFont.load_default(size=11)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
        legend_font = font

    keyposes = [compute_keypose(s, hold_px, dims) for s in states]

    # Shared global palette — per-frame palettes are what balloon GIF size.
    # Seed it with the figure colours so they survive quantisation.
    pal_src = base.copy()
    seed = ImageDraw.Draw(pal_src)
    # large swatches so median-cut gives these colours their own buckets
    seed_colours = (ACCENT, OUTLINE, "#ffffff", BODY) + tuple(LIMB_COLOURS.values())
    sw = max(30, out_width // 10)
    for k, col in enumerate(seed_colours):
        seed.rectangle([k * sw, 0, (k + 1) * sw - 1, sw - 1], fill=col)
    palette_img = pal_src.quantize(colors=128)

    frames = []

    def add_frame(pts, label, ring=None, ring_t=0.0, ring_colour=ACCENT):
        frame = base.copy()
        draw = ImageDraw.Draw(frame)
        if ring is not None:
            rad = 14 + 8 * math.sin(ring_t * math.pi)
            x, y = ring
            draw.ellipse([x - rad, y - rad, x + rad, y + rad],
                         outline=ring_colour, width=3)
        draw_figure(draw, pts, dims, line_w)
        _draw_label(draw, label, font, out_width)
        _draw_legend(draw, legend_font, out_width)
        frames.append(frame.quantize(palette=palette_img, dither=Image.Dither.NONE))

    # intro: start position
    for _ in range(INTRO_FRAMES):
        add_frame(keyposes[0], _label_for(sequence[0]))

    # animate each move
    n = min(len(keyposes), len(sequence))
    for i in range(1, n):
        a, b = keyposes[i - 1], keyposes[i]
        changed = [j for j in JOINTS if _dist(a[j], b[j]) > 2]
        label = _label_for(sequence[i])

        # ring on the destination hold, coloured by the moving limb
        ring = None
        joint = _limb_joint(sequence[i]["limb"])
        ring_colour = LIMB_COLOURS.get(joint, ACCENT)
        if sequence[i]["hold"] is not None and sequence[i]["hold"] in hold_px:
            ring = hold_px[sequence[i]["hold"]]

        for f in range(FRAMES_PER_MOVE):
            t = _ease((f + 1) / FRAMES_PER_MOVE)
            pts = {}
            for j in JOINTS:
                if j in changed:
                    pos = _lerp(a[j], b[j], t)
                    # arc lift, perpendicular to travel, biased upward
                    dx, dy = b[j][0] - a[j][0], b[j][1] - a[j][1]
                    seg = math.hypot(dx, dy) or 1.0
                    perp = (-dy / seg, dx / seg)
                    if perp[1] > 0:
                        perp = (-perp[0], -perp[1])
                    lift = 0.18 * seg * math.sin(t * math.pi)
                    pts[j] = (pos[0] + perp[0] * lift, pos[1] + perp[1] * lift)
                else:
                    pts[j] = a[j]
            add_frame(pts, label, ring, t, ring_colour)

        for _ in range(HOLD_FRAMES):
            add_frame(b, label)

    # linger on the final pose
    for _ in range(HOLD_FRAMES * 2):
        add_frame(keyposes[n - 1], "Send!")

    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=frame_ms, loop=0, optimize=True,
    )
    buf.seek(0)
    return buf
