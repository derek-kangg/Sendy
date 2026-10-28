from route_graph import build_reachability_graph
from beta_animation import render_beta_gif
from climbing_knowledge import coaching_knowledge
from sequence_engine import (build_holds_description, generate_sequence_iteratively,
                             friendly_api_error, MAX_MOVES)
from hold_detection import (detect_and_validate_holds, redraw_annotation,
                            marker_metrics as _marker_metrics, DetectionError, NoHoldsFound)
from image_ingest import prepare_upload, ImageRejected
import guardrails
from guardrails import LimitExceeded
from PIL import Image, ImageDraw
import streamlit as st
import anthropic
from dotenv import load_dotenv
import os
import base64
import io
import json
import shutil
import re
import html
import hashlib
import hmac
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("climbai.app")

DEV_MODE          = os.getenv("CLIMBAI_DEV", "").strip().lower() in ("1", "true", "yes")
ACCESS_CODE       = os.getenv("CLIMBAI_ACCESS_CODE", "").strip()   # empty = open to everyone
CLICK_IMAGE_MAX_W = 900          # what the tap-to-select component ships to the phone
VISITOR_TTL_S     = 24 * 3600    # per-visitor working folders older than this are purged
JOB_TTL_S         = 3600         # finished generation jobs are kept this long for reattach


def get_api_key():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        try:
            key = st.secrets["ANTHROPIC_API_KEY"]
        except (KeyError, FileNotFoundError):
            key = None
    return key


@st.cache_resource(show_spinner=False)
def get_client():
    """One metered Anthropic client shared by every session (connection
    pooling + one spend ledger). Short timeouts: a hung call must not block
    a climber for the SDK's default ten minutes."""
    return guardrails.wrap_client(anthropic.Anthropic(api_key=get_api_key(), timeout=60.0, max_retries=2))


@st.cache_resource(show_spinner=False)
def _job_store():
    """Background generation jobs, process-wide. Module globals would be
    re-created on every Streamlit rerun; a cached resource survives. Two
    workers bound RAM and API concurrency; a third climber waits in line."""
    return {"jobs": {}, "lock": threading.Lock(),
            "executor": ThreadPoolExecutor(max_workers=2, thread_name_prefix="beta")}


_APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROUTES_ROOT      = os.path.join(_APP_DIR, "saved_routes")
TEST_ROUTES_ROOT = os.path.join(_APP_DIR, "test_routes")


def _visitor_id():
    """A random id per browser, carried in the URL so a refresh or a dropped
    connection finds the same working folder. Nothing is ever listed across
    ids: one visitor cannot see another's photos."""
    vid = st.session_state.get("visitor_id")
    if not vid:
        qp = st.query_params.get("v")
        vid = qp if isinstance(qp, str) and re.fullmatch(r"[0-9a-f]{16}", qp) else uuid.uuid4().hex[:16]
        st.session_state["visitor_id"] = vid
    if st.query_params.get("v") != vid:
        st.query_params["v"] = vid
    return vid


VISITOR_ID  = _visitor_id()
VISITOR_DIR = os.path.join(ROUTES_ROOT, VISITOR_ID)
SAVE_DIR    = os.path.join(VISITOR_DIR, "last")        # autosave for this visitor only
UPLOAD_PATH = os.path.join(VISITOR_DIR, "upload.jpg")  # the normalised photo everything reads


def _slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")


def _atomic_write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _purge_old_visitor_dirs():
    """Disk is ephemeral and shared by every visitor; keep it bounded."""
    if not os.path.isdir(ROUTES_ROOT):
        return
    cutoff = time.time() - VISITOR_TTL_S
    for name in os.listdir(ROUTES_ROOT):
        path = os.path.join(ROUTES_ROOT, name)
        try:
            if name != VISITOR_ID and os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def save_route(target_dir=None):
    """Persist the current route so it can be resumed later without
    re-running hold detection or beta generation (no API cost)."""
    target = target_dir or SAVE_DIR
    os.makedirs(target, exist_ok=True)
    ss = st.session_state
    data = {
        "holds":        ss.get("holds"),
        "start_holds":  ss.get("start_holds", []),
        "sequence":     ss.get("sequence"),
        "states":       ss.get("states"),
        "instructions": ss.get("instructions"),
        "beta_status":  ss.get("beta_status"),
        "beta_reason":  ss.get("beta_reason"),
    }
    _atomic_write(os.path.join(target, "route.json"), json.dumps(data).encode("utf-8"))
    if ss.get("annotated_image") is not None:
        buf = io.BytesIO()
        ss["annotated_image"].save(buf, format="PNG")
        _atomic_write(os.path.join(target, "annotated.png"), buf.getvalue())
    src = ss.get("tmp_path")
    original = os.path.join(target, "original.jpg")
    if src and os.path.exists(src) and os.path.abspath(src) != os.path.abspath(original):
        shutil.copyfile(src, original + ".tmp")
        os.replace(original + ".tmp", original)
    if ss.get("beta_gif"):
        _atomic_write(os.path.join(target, "beta.gif"), ss["beta_gif"])
    os.utime(VISITOR_DIR, None)


def list_saved_routes():
    """Names of this visitor's saved routes (the autosave is not listed)."""
    if not os.path.isdir(VISITOR_DIR):
        return []
    return [n for n in sorted(os.listdir(VISITOR_DIR))
            if n != "last" and os.path.exists(os.path.join(VISITOR_DIR, n, "route.json"))]


def load_route(source_dir=None):
    """Restore a saved route into session state. Returns True on success."""
    source = source_dir or SAVE_DIR
    try:
        with open(os.path.join(source, "route.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        annotated = Image.open(os.path.join(source, "annotated.png")).convert("RGB")
    except Exception:
        return False
    ss = st.session_state
    _clear_route_state()
    ss["holds"]           = data.get("holds") or []
    ss["start_holds"]     = data.get("start_holds") or []
    ss["annotated_image"] = annotated
    ss["base_image"]      = annotated
    ss["tmp_path"]        = os.path.join(source, "original.jpg")
    ss["current_step"]    = 0
    if data.get("sequence"):
        ss["sequence"] = data["sequence"]
    if data.get("states"):
        ss["states"] = data["states"]
    if data.get("instructions"):
        ss["instructions"] = data["instructions"]
    if data.get("beta_status"):
        ss["beta_status"] = data["beta_status"]
        ss["beta_reason"] = data.get("beta_reason", "")
    gif_path = os.path.join(source, "beta.gif")
    if os.path.exists(gif_path):
        with open(gif_path, "rb") as f:
            ss["beta_gif"] = f.read()
    return True


def _clear_route_state():
    """Forget everything derived from the current photo (not the photo itself)."""
    for key in ("holds", "annotated_image", "base_image", "start_holds", "sequence", "states",
                "instructions", "beta_status", "beta_reason", "beta_gif", "beta_feedback",
                "beta_error", "current_step", "last_click", "job_id"):
        st.session_state.pop(key, None)
    try:
        del st.query_params["job"]
    except KeyError:
        pass


def export_test_route(slug, colour, wall_angle, difficulty):
    """Developer tool (CLIMBAI_DEV=1): write the current route into
    test_routes/<slug>/ in the evaluation format."""
    target = os.path.join(TEST_ROUTES_ROOT, slug)
    os.makedirs(target, exist_ok=True)
    ss = st.session_state

    tmp = ss.get("tmp_path")
    if tmp and os.path.exists(tmp):
        shutil.copyfile(tmp, os.path.join(target, "wall.jpg"))

    holds = ss.get("holds") or []
    seq   = ss.get("sequence") or []

    hold_lines = "\n".join(
        f"- Hold {h['number']}: {h.get('hold_type', '?')}, {h.get('size', '?')}"
        + (f", {h['orientation']}" if h.get("orientation") not in (None, "unknown", "top") else "")
        + f", x={h['x']}, y={h['y']}"
        for h in holds
    )
    ai_lines = "\n".join(
        f"{m['move_number']}. {m['limb']} -> "
        + (f"hold {m['hold']}" if m["hold"] is not None else "wall")
        + f" ({m['action']}) - {m.get('cue', '')}"
        for m in seq
    )

    md = f"""# {slug}

- Gym / wall:
- Colour: {colour}
- Grade:
- Wall angle: {wall_angle}
- Level used in app: {difficulty}
- Start:
- Finish:

## Detected holds

{hold_lines}

## AI suggested beta (for reference)

{ai_lines}

## My verified beta (fill this in after climbing)

START:
1.

## Notes

"""
    with open(os.path.join(target, "route.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(target, "route_data.json"), "w", encoding="utf-8") as f:
        json.dump({
            "holds": holds, "ai_sequence": seq, "colour": colour,
            "wall_angle": wall_angle, "difficulty": difficulty,
        }, f, indent=2)
    return os.path.relpath(target, _APP_DIR)


def format_sequence_as_text(sequence, hold_descriptions):
    lines = []
    for move in sequence:
        num      = move["move_number"]
        limb     = move["limb"]
        hold     = move["hold"]
        cue      = move.get("cue", "")
        hold_str = f"Hold {hold}" if hold is not None else "wall"

        if num == 0:
            lines.append(f"**Start:** {cue}")
        else:
            lines.append(f"**Move {num}:** {limb} → {hold_str} — {cue}")

    lines.append("\n**Hold analysis**\n")
    lines.append(hold_descriptions)
    return "\n\n".join(lines)


def analyze_user_beta(client, annotated_image, holds, user_beta, height_cm, difficulty, wall_angle):
    """Rate My Beta: the climber describes their own sequence and gets coaching feedback."""
    buf = io.BytesIO()
    annotated_image.save(buf, format="JPEG")
    image_data = base64.b64encode(buf.getvalue()).decode("utf-8")

    holds_description = build_holds_description(holds)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        timeout=60.0,
        system=f"""You are an experienced bouldering coach reviewing a climber's beta (their planned sequence). Be supportive but honest — like a good coach at the gym.

Your feedback should cover:
1. WHAT WORKS — parts of their beta that are solid, and why
2. WATCH OUT FOR — risks or inefficiencies (balance issues, skipped feet, over-gripping, wrong grip for the hold type)
3. SUGGESTIONS — at most 2-3 concrete improvements, only where they genuinely help. If their beta is good, say so — do not invent problems.

Refer to holds by their numbers. Use precise technique vocabulary where it fits (grip choice for the hold type, drop knee vs backstep, deadpoint vs dyno). Keep it conversational and under 300 words. Never rewrite their whole sequence — coach the beta they brought you.

{coaching_knowledge()}""",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                {"type": "text", "text": f"""Here is the climbing wall with numbered holds.

{holds_description}

CLIMBER: {height_cm}cm, {difficulty} level
WALL: {wall_angle}

MY BETA:
{user_beta}

Please review my beta."""}
            ]
        }]
    )
    return response.content[0].text


def draw_move_overlay(base_image, holds, sequence, current_step):
    hold_map = {h["number"]: h for h in holds}
    image = base_image.copy()
    draw  = ImageDraw.Draw(image)

    # One colour per limb, matching the animated figure:
    # LH yellow, RH orange, LF green, RF blue
    COLOURS = {
        "right hand":       "#fb923c",
        "left hand":        "#facc15",
        "both hands":       "#fb923c",
        "right foot":       "#38bdf8",
        "left foot":        "#4ade80",
        "both feet":        "#38bdf8",
        "flag right foot":  "#38bdf8",
        "flag left foot":   "#4ade80",
        "smear right foot": "#38bdf8",
        "smear left foot":  "#4ade80",
        "swap right foot":  "#38bdf8",
        "swap left foot":   "#4ade80",
    }

    # marker sizes scale with resolution (fixed sizes vanish on phone photos)
    s = max(1.0, image.width / 770.0)
    ring_w = max(2, int(2 * s))
    mr, mfont = _marker_metrics(image.width)

    for move in sequence[:current_step]:
        if move["hold"] is None:
            continue
        hold_num = move["hold"]
        if hold_num not in hold_map:
            continue
        hx, hy = hold_map[hold_num]["x"], hold_map[hold_num]["y"]
        colour = COLOURS.get(move["limb"], "#ffffff")
        r = int(18 * s)
        draw.ellipse([hx-r, hy-r, hx+r, hy+r], outline=colour, width=ring_w)

    current_move = sequence[current_step]
    if current_move["hold"] is not None:
        hold_num = current_move["hold"]
        if hold_num in hold_map:
            hx, hy = hold_map[hold_num]["x"], hold_map[hold_num]["y"]
            colour = COLOURS.get(current_move["limb"], "#ffffff")
            r_out, r_in = int(35 * s), int(25 * s)
            draw.ellipse([hx-r_out, hy-r_out, hx+r_out, hy+r_out], outline=colour, width=ring_w)
            draw.ellipse([hx-r_in, hy-r_in, hx+r_in, hy+r_in], fill=colour)
            draw.text((hx, hy), str(current_move["move_number"]), fill="black", font=mfont, anchor="mm")

    return image


def click_image_for(annotated_image, start_holds):
    """What the tap-to-select component shows: a phone-sized copy of the
    annotated photo with the chosen start holds ringed. Shipping the
    full-size image as PNG on every tap was tens of megabytes."""
    w, h = annotated_image.size
    cw = min(w, CLICK_IMAGE_MAX_W)
    scale = cw / w
    img = annotated_image if cw == w else annotated_image.resize((cw, round(h * scale)), Image.Resampling.LANCZOS)
    if start_holds:
        img = img.copy()
        draw = ImageDraw.Draw(img)
        mr, _ = _marker_metrics(w)
        r = int(mr * scale * 1.6)
        for hold in start_holds:
            x, y = hold["x"] * scale, hold["y"] * scale
            draw.ellipse([x - r, y - r, x + r, y + r], outline="#4ade80", width=max(3, int(4 * scale)))
    return img


# ---- Background beta generation ----
# Generation is 10-60 sequential model calls (30-90 s). Running it inside the
# Streamlit script meant any rerun, phone lock or dropped WebSocket killed it
# and threw away every move already paid for. A worker thread owns the job;
# the page polls it and can reattach to it by id after a reconnect.

def _start_generation_job(client, holds, start_holds, finish_hold, height_cm, difficulty,
                          wall_angle, finish_style, extra_notes, image_size, wall_height_cm):
    store = _job_store()
    now = time.time()
    with store["lock"]:
        for jid in [j for j, v in store["jobs"].items() if now - v["created"] > JOB_TTL_S]:
            del store["jobs"][jid]
        job_id = uuid.uuid4().hex[:12]
        job = {"progress": 0, "total": MAX_MOVES, "result": None, "error": None,
               "started": False, "done": False, "created": now, "visitor": VISITOR_ID}
        store["jobs"][job_id] = job

    holds = [dict(h) for h in holds]
    start_holds = [dict(h) for h in start_holds]
    image_width, image_height = image_size
    graph = build_reachability_graph(holds, image_height, image_width, height_cm, wall_height_cm=wall_height_cm)
    hold_descriptions = build_holds_description(holds)

    def on_progress(n, total):
        job["progress"], job["total"] = n, total

    def work():
        job["started"] = True
        try:
            job["result"] = generate_sequence_iteratively(
                client, hold_descriptions, holds, graph, start_holds,
                height_cm, difficulty, wall_angle, finish_style, extra_notes,
                image_height=image_height, progress_callback=on_progress,
                wall_height_cm=wall_height_cm, finish_hold=finish_hold,
            )
        except Exception:
            log.exception("generation job %s crashed", job_id)
            job["error"] = "Something went wrong while generating the beta — please try again."
        finally:
            job["done"] = True

    store["executor"].submit(work)
    return job_id


def _consume_job(job):
    ss = st.session_state
    res = job["result"]
    if job["error"] or res is None:
        ss["beta_error"] = job["error"] or "Beta generation did not finish — please try again."
        return
    if res.status == "error" and len(res.sequence) <= 1:
        ss["beta_error"] = res.reason
        return
    hold_descriptions = build_holds_description(ss["holds"])
    ss["instructions"]  = format_sequence_as_text(res.sequence, hold_descriptions)
    ss["sequence"]      = res.sequence
    ss["states"]        = res.states
    ss["beta_status"]   = res.status
    ss["beta_reason"]   = res.reason
    ss["current_step"]  = 0
    ss["base_image"]    = ss["annotated_image"]
    ss.pop("beta_gif", None)
    save_route()


@st.fragment(run_every=1.0)
def _generation_progress():
    store = _job_store()
    job = store["jobs"].get(st.session_state.get("job_id"))
    if job is None:
        st.session_state.pop("job_id", None)
        return
    if job["done"]:
        _consume_job(job)
        st.session_state.pop("job_id", None)
        try:
            del st.query_params["job"]
        except KeyError:
            pass
        st.rerun()
    n, total = job["progress"], job["total"]
    if not job["started"]:
        st.progress(0.0, text="Waiting for a free slot — another climber's beta is generating right now.")
    else:
        st.progress(min(n / total, 1.0), text=f"Working out move {n} (up to {total}) — about 3–5 seconds per move")
    st.caption("You can lock your phone or switch apps — this page keeps working and shows the beta when you come back.")


# ---- UI ----

st.set_page_config(page_title="ClimbAI", page_icon="🧗", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --bg: #0f1011;
    --surface: #1a1b1e;
    --border: #26282c;
    --text: #ececec;
    --muted: #9b9ea6;
    --accent: #fb923c;
}

html, body, [class*="st-"], .stMarkdown, input, textarea, select {
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}

/* Restore Streamlit's icon font — icons are ligatures and render as raw words
   (e.g. "upload") if the font-family override reaches them */
[data-testid="stIconMaterial"], [class*="material-symbols"] {
    font-family: 'Material Symbols Rounded' !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

/* Layout */
.block-container { max-width: 720px; padding-top: 4rem; padding-bottom: 6rem; }

/* Hero */
.climb-hero { margin-bottom: 3rem; }
.climb-hero .wordmark {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem; font-weight: 700; letter-spacing: -0.045em;
    color: var(--text); margin: 0; line-height: 1;
}
.climb-hero .wordmark .accent { color: var(--accent); }
.climb-hero .tagline {
    color: var(--muted); font-size: 1.02rem; margin-top: 0.9rem;
    font-weight: 400; max-width: 30rem; line-height: 1.55;
}

/* Section labels — number, title, keyline */
.section-label {
    display: flex; align-items: center; gap: 0.75rem;
    margin: 3rem 0 0.3rem;
}
.section-label .num {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--accent); font-size: 0.78rem; font-weight: 600; letter-spacing: 0.1em;
}
.section-label .title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.12rem; font-weight: 600; color: var(--text); letter-spacing: -0.01em;
    white-space: nowrap;
}
.section-label::after {
    content: ""; flex: 1; height: 1px; background: var(--border);
}
.section-caption { color: var(--muted); font-size: 0.87rem; margin: 0.15rem 0 0.7rem; line-height: 1.5; }
.accent-text { color: var(--accent); font-weight: 600; }

/* Buttons */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    border-radius: 10px; font-weight: 600; padding: 0.7rem 1rem;
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    transition: border-color 0.15s ease, color 0.15s ease, filter 0.15s ease;
}
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stButton > button[kind="primary"] {
    background: var(--accent); color: var(--bg); border: none;
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.1); color: var(--bg); }

/* Inputs */
.stSelectbox > div > div, .stNumberInput > div > div, .stTextArea textarea {
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"] {
    border-radius: 12px; border: 1px dashed var(--border) !important;
    background: var(--surface) !important;
}
/* Hide drag-and-drop instructions — single clean button, mobile friendly */
[data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
[data-testid="stFileUploaderDropzone"] {
    justify-content: center; align-items: center; padding: 0.6rem; gap: 0;
}
/* The upload icon is a SIBLING of the button inside the dropzone —
   hide icons anywhere in the dropzone, at any depth */
[data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"],
[data-testid="stFileUploaderDropzone"] [class*="material-symbols"],
[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stFileUploaderDropzone"] i {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] button {
    font-family: 'Space Grotesk', sans-serif !important;
    border-radius: 8px; font-weight: 600;
    width: 100%; padding: 0.55rem 1rem;
    display: flex; align-items: center; justify-content: center; gap: 0;
    text-align: center;
}
[data-testid="stFileUploaderDropzone"] button > * { margin: 0 !important; }
/* The button sits inside an auto-width wrapper span — stretch it so the
   full-width button actually spans the dropzone */
[data-testid="stFileUploaderDropzone"] > span { display: block; width: 100%; }
[data-testid="stFileUploaderDropzone"] button * { margin: 0 !important; }

/* Alerts */
.stAlert { border-radius: 10px; }

/* Move card */
.move-card {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: 10px;
    padding: 0.95rem 1.2rem; margin: 0.4rem 0 1rem;
}
.move-card .move-num {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--accent); font-weight: 600; font-size: 0.72rem;
    letter-spacing: 0.12em; text-transform: uppercase;
}
.move-card .move-text {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text); font-size: 1.05rem; margin-top: 4px; font-weight: 600;
}
.move-card .move-cue { color: var(--muted); font-size: 0.88rem; margin-top: 3px; }

/* Footer */
.climb-footer {
    font-family: 'Space Grotesk', sans-serif;
    color: #55585e; font-size: 0.72rem; text-align: center; margin-top: 5rem;
    text-transform: uppercase; letter-spacing: 0.18em;
}

hr { margin: 1.6rem 0 !important; opacity: 0.25; }
</style>

<div class="climb-hero">
    <h1 class="wordmark">climb<span class="accent">ai</span></h1>
    <p class="tagline">Route reading, assisted. Upload a wall photo — get a suggested beta and coaching feedback on your own.</p>
</div>
""", unsafe_allow_html=True)


def section_header(num, title, caption=None):
    st.markdown(f"""<div class="section-label"><span class="num">{num}</span><span class="title">{title}</span></div>""", unsafe_allow_html=True)
    if caption:
        st.markdown(f"""<p class="section-caption">{caption}</p>""", unsafe_allow_html=True)


if not get_api_key():
    st.error("ClimbAI isn't configured yet — the owner needs to set ANTHROPIC_API_KEY.")
    st.stop()


def _unlocked():
    """Optional access code for a private beta: one key serves everyone, so
    the link is not public until the owner says so. Remembered per visitor
    link, so a refresh does not ask again."""
    if not ACCESS_CODE or st.session_state.get("unlocked"):
        return True
    if os.path.exists(os.path.join(VISITOR_DIR, ".unlocked")):
        st.session_state["unlocked"] = True
        return True
    return False


if not _unlocked():
    st.markdown("""<p class="section-caption">ClimbAI is in a private beta. Enter the access code you were given.</p>""", unsafe_allow_html=True)
    code = st.text_input("Access code", type="password", label_visibility="collapsed", placeholder="Access code")
    if st.button("Enter", type="primary"):
        if hmac.compare_digest(code.strip(), ACCESS_CODE):
            os.makedirs(VISITOR_DIR, exist_ok=True)
            with open(os.path.join(VISITOR_DIR, ".unlocked"), "w") as f:
                f.write("1")
            st.session_state["unlocked"] = True
            st.rerun()
        else:
            st.error("That code didn't match.")
    st.stop()

anthropic_client = get_client()

if os.getenv("CLIMBAI_DISABLED", "").strip().lower() in ("1", "true", "yes"):
    st.info("ClimbAI's AI features are switched off right now — routes you already generated still work.")

# A generation started before a refresh or a dropped connection: pick it back up.
_qp_job = st.query_params.get("job")
if _qp_job and "job_id" not in st.session_state and _qp_job in _job_store()["jobs"]:
    st.session_state["job_id"] = _qp_job

section_header("01", "The route", "A clear, straight-on photo works best.")

uploaded_file = st.file_uploader("Climbing wall photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
# Only mount the camera widget on request — mounted, it asks for camera
# permission the moment the page loads, even inside a collapsed expander.
if st.toggle("Take a photo with the camera instead"):
    camera_shot = st.camera_input("Camera", label_visibility="collapsed")
    if uploaded_file is None and camera_shot is not None:
        uploaded_file = camera_shot
st.markdown("""<p class="section-caption">Photos stay on the server only for your session and are deleted within a day. Try to keep other climbers out of the frame.</p>""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    colour = st.selectbox("Route colour",
        ["Black", "Blue", "Red", "Green", "Orange", "Pink", "White", "Yellow", "Purple"])
with col2:
    wall_angle = st.selectbox("Wall angle",
        ["Slab (less than vertical)", "Vertical", "Slight overhang", "Steep overhang", "Roof"])

col3, col4 = st.columns(2)
with col3:
    wall_height_m = st.number_input("Wall height (m)", min_value=2.5, max_value=6.0, value=4.0, step=0.5,
        help="Roughly how much wall the photo shows, top to bottom. This sets the scale for what's in reach.")
with col4:
    finish_style = st.selectbox("Finish marking",
        ["TOP label", "Tape on hold", "Top-out (both hands on top)", "Not marked"])
wall_height_cm = int(wall_height_m * 100)

section_header("02", "About you")

col5, col6 = st.columns(2)
with col5:
    difficulty = st.selectbox("Experience level",
        ["Beginner", "Intermediate", "Advanced"])
with col6:
    height_cm = st.number_input("Height (cm)", min_value=140, max_value=220, value=170)

extra_notes = st.text_area("Notes (optional)",
    placeholder="e.g. 'There is a big move in the middle'",
    height=80)

upload_ok = False
if uploaded_file is not None:
    data = uploaded_file.getvalue()
    digest = hashlib.sha1(data).hexdigest()
    if st.session_state.get("upload_digest") != digest:
        # A new photo: normalise it once (orientation, size) and forget the old route.
        try:
            os.makedirs(VISITOR_DIR, exist_ok=True)
            prepare_upload(data, UPLOAD_PATH)
        except ImageRejected as e:
            st.error(str(e))
        else:
            _clear_route_state()
            st.session_state.pop("detect_cache", None)
            st.session_state["upload_digest"] = digest
            st.session_state["tmp_path"] = UPLOAD_PATH
    upload_ok = st.session_state.get("upload_digest") == digest

    section_header("03", "Detect holds")

    if upload_ok and st.button("Detect holds", use_container_width=True, type="primary"):
        try:
            # Same photo + same colour again (a second tap, or toggling back
            # after trying another colour) costs nothing.
            cache = st.session_state.setdefault("detect_cache", {})
            cached = cache.get((digest, colour))
            if cached:
                annotated_image, holds = cached[0], [dict(h) for h in cached[1]]
            else:
                guardrails.LEDGER.check("detect", VISITOR_ID)
                with st.spinner("Detecting holds and analysing types..."):
                    guardrails.LEDGER.count("detect", VISITOR_ID)
                    _purge_old_visitor_dirs()
                    annotated_image, holds = detect_and_validate_holds(anthropic_client, UPLOAD_PATH, colour)
                cache[(digest, colour)] = (annotated_image, [dict(h) for h in holds])
            _clear_route_state()
            st.session_state["annotated_image"] = annotated_image
            st.session_state["holds"]           = holds
            st.session_state["start_holds"]     = []
            st.session_state["current_step"]    = 0
            save_route()
        except LimitExceeded as e:
            st.warning(str(e))
        except NoHoldsFound as e:
            st.warning(str(e))
        except DetectionError as e:
            st.error(str(e))
        except anthropic.APIError as e:
            log.warning("detection API error: %s", e)
            st.error(friendly_api_error(e))
        except Exception:
            log.exception("detection failed")
            st.error("Hold detection failed unexpectedly — please try again.")

elif "holds" not in st.session_state:
    # No photo this session: bring back this visitor's last route (same link,
    # e.g. after a refresh), or offer their saved library.
    if not st.session_state.get("restored_once"):
        st.session_state["restored_once"] = True
        if load_route(SAVE_DIR):
            st.rerun()
    saved = list_saved_routes()
    if saved:
        section_header("03", "Detect holds")
        st.markdown("""<p class="section-caption">No photo uploaded — load one of your saved routes instead. No re-detection or regeneration needed.</p>""", unsafe_allow_html=True)
        col_pick, col_load = st.columns([2, 1])
        with col_pick:
            chosen_route = st.selectbox("Saved routes", saved, label_visibility="collapsed")
        with col_load:
            if st.button("Load route", use_container_width=True, type="primary"):
                if load_route(os.path.join(VISITOR_DIR, _slugify(chosen_route))):
                    st.rerun()
                else:
                    st.warning("Couldn't load that route — upload a photo instead.")

if "annotated_image" in st.session_state and st.session_state.get("holds"):
    holds           = st.session_state["holds"]
    annotated_image = st.session_state["annotated_image"]
    if uploaded_file is None:
        section_header("03", "Detect holds")
        col_msg, col_reset = st.columns([3, 1])
        with col_msg:
            st.markdown("""<p class="section-caption" style="margin-top:0.6rem;">Restored your last route. Upload a new photo above to start another.</p>""", unsafe_allow_html=True)
        with col_reset:
            if st.button("Start over", use_container_width=True):
                _clear_route_state()
                st.session_state.pop("upload_digest", None)
                shutil.rmtree(SAVE_DIR, ignore_errors=True)
                st.rerun()

    st.markdown(f"""<p class="section-caption" style="margin-top:0.8rem;">Found <span class="accent-text">{len(holds)}</span> {colour.lower()} holds. Tap the start hold(s) below — one if both hands start together, two if they start apart. Tap again to deselect. Detection wrong somewhere? Switch the tap mode to add or remove holds.</p>""", unsafe_allow_html=True)

    click_mode = st.radio(
        "Click mode",
        ["Select start holds", "Add hold", "Remove hold"],
        horizontal=True,
        label_visibility="collapsed",
        key="click_mode",
    )

    from streamlit_image_coordinates import streamlit_image_coordinates
    start_holds_now = st.session_state.get("start_holds", [])
    click_img = click_image_for(annotated_image, start_holds_now)
    # JPEG, not the component's default uncompressed PNG: ~0.4 MB instead of
    # ~20 MB per render on a phone photo.
    value = streamlit_image_coordinates(click_img, key="hold_click", use_column_width="always",
                                        image_format="JPEG")

    def _click_id(v):
        # The component stamps each tap; falling back to (x, y) would ignore a
        # second tap on the same spot (deselect then reselect the same hold).
        return v.get("unix_time") or (v["x"], v["y"])

    # The component replays its last click on every rerun. Switching the tap
    # mode must not re-apply that old tap as an add/remove — swallow it.
    if st.session_state.get("prev_click_mode") != click_mode:
        st.session_state["prev_click_mode"] = click_mode
        st.session_state["last_click"] = _click_id(value) if value else None
        value = None

    if value is not None:
        # The component reports clicks in DISPLAYED pixels (plus the displayed
        # size) — rescale into original image space or taps on phones, where
        # the image is shrunk to fit, would land on the wrong holds.
        disp_w = value.get("width") or click_img.width
        disp_h = value.get("height") or click_img.height
        click_x = value["x"] * annotated_image.width / disp_w
        click_y = value["y"] * annotated_image.height / disp_h

        # Match threshold scales with image size (fat-finger friendly)
        threshold = max(40, annotated_image.width * 0.06)

        closest_hold = None
        closest_dist = float("inf")
        for h in holds:
            dist = ((h["x"] - click_x) ** 2 + (h["y"] - click_y) ** 2) ** 0.5
            if dist < closest_dist:
                closest_dist = dist
                closest_hold = h

        last_click = st.session_state.get("last_click", None)
        new_click  = _click_id(value)
        is_new_click = last_click != new_click

        def _drop_beta():
            """Anything downstream of the holds/start is stale now."""
            for key in ("sequence", "states", "instructions", "beta_status", "beta_reason", "beta_gif", "beta_error"):
                st.session_state.pop(key, None)

        def _holds_edited():
            """After add/remove: redraw annotation, drop stale downstream state."""
            st.session_state["annotated_image"] = redraw_annotation(
                st.session_state["tmp_path"], st.session_state["holds"])
            _drop_beta()
            save_route()

        if is_new_click and click_mode == "Add hold":
            st.session_state["last_click"] = new_click
            new_num = max((h["number"] for h in holds), default=0) + 1
            half = int(max(20, annotated_image.width * 0.02))
            holds.append({
                "number": new_num,
                "x": int(click_x), "y": int(click_y),
                "size": "medium (likely handhold)",
                "hold_type": "unknown", "best_use": "both", "orientation": "unknown",
                "bx": int(click_x) - half, "by": int(click_y) - half,
                "bw": half * 2, "bh": half * 2,
            })
            st.session_state["holds"] = holds
            _holds_edited()
            st.toast(f"Added Hold {new_num}")
            st.rerun()

        elif is_new_click and click_mode == "Remove hold":
            st.session_state["last_click"] = new_click
            if closest_hold and closest_dist < threshold:
                st.session_state["holds"] = [h for h in holds if h["number"] != closest_hold["number"]]
                st.session_state["start_holds"] = [
                    h for h in st.session_state.get("start_holds", [])
                    if h["number"] != closest_hold["number"]
                ]
                _holds_edited()
                st.toast(f"Removed Hold {closest_hold['number']}")
                st.rerun()
            else:
                st.toast("No hold there — tap closer to a marker to remove it.")

        elif is_new_click and click_mode == "Select start holds":
            st.session_state["last_click"] = new_click
            if closest_hold and closest_dist < threshold:
                start_holds = list(st.session_state.get("start_holds", []))
                hold_nums   = [h["number"] for h in start_holds]
                if closest_hold["number"] in hold_nums:
                    start_holds = [h for h in start_holds if h["number"] != closest_hold["number"]]
                    st.toast(f"Deselected Hold {closest_hold['number']}")
                elif len(start_holds) >= 2:
                    st.toast("Two start holds max — deselect one first.")
                    start_holds = None
                else:
                    start_holds.append(closest_hold)
                    st.toast(f"Selected Hold {closest_hold['number']} as a start hold")
                if start_holds is not None:
                    st.session_state["start_holds"] = start_holds
                    if "sequence" in st.session_state:
                        _drop_beta()
                        st.toast("Start changed — generate the beta again.")
                    save_route()
                    st.rerun()
            else:
                st.toast("No hold there — tap closer to a marker.")

    current_start_holds = st.session_state.get("start_holds", [])
    if current_start_holds:
        hold_nums = [str(h["number"]) for h in current_start_holds]
        st.success(f"Start holds: Hold {', Hold '.join(hold_nums)}")
    else:
        st.info("Tap the start hold(s) on the image above.")

    section_header("04", "Suggested beta")

    highest = min(holds, key=lambda h: h["y"])["number"]
    finish_options = [f"Highest hold (Hold {highest})"] + [f"Hold {n}" for n in sorted(h["number"] for h in holds)]
    finish_choice = st.selectbox("Finish hold", finish_options,
        help="Which hold is the top of the route? Defaults to the highest detected hold.")
    finish_hold = highest if finish_choice.startswith("Highest") else int(finish_choice.split()[1])

    if st.session_state.get("beta_error"):
        st.error(st.session_state.pop("beta_error"))

    job_running = st.session_state.get("job_id") in _job_store()["jobs"]
    if job_running:
        _generation_progress()
    elif st.button("Generate suggested beta", use_container_width=True, type="primary"):
        current_start_holds = st.session_state.get("start_holds", [])
        if not current_start_holds:
            st.warning("Tap the start holds on the image before generating a beta.")
        else:
            try:
                guardrails.LEDGER.check("generate", VISITOR_ID)
                guardrails.LEDGER.count("generate", VISITOR_ID)
                image_size = Image.open(st.session_state["tmp_path"]).size
                job_id = _start_generation_job(
                    anthropic_client, st.session_state["holds"], current_start_holds, finish_hold,
                    height_cm, difficulty, wall_angle, finish_style, extra_notes,
                    image_size, wall_height_cm,
                )
                st.session_state["job_id"] = job_id
                st.query_params["job"] = job_id
                st.rerun()
            except LimitExceeded as e:
                st.warning(str(e))
            except Exception:
                log.exception("could not start generation")
                st.error("Couldn't start the beta generation — please try again.")

if "instructions" in st.session_state:
    if st.session_state.get("beta_status") and st.session_state["beta_status"] != "complete":
        st.warning(f"This beta is incomplete — {st.session_state.get('beta_reason', '')} "
                   "Showing the moves it did work out; generate again for another attempt.")
    st.markdown("""<p class="section-caption" style="margin-top:0.8rem;">A starting point — adapt it to your body, strengths, and style. Even experienced climbers refine beta on the wall.</p>""", unsafe_allow_html=True)
    st.markdown(st.session_state["instructions"])

if "sequence" in st.session_state and st.session_state["sequence"]:
    sequence     = st.session_state["sequence"]
    current_step = min(st.session_state.get("current_step", 0), len(sequence) - 1)

    section_header("05", "Walk it through", "Step through each move on the wall.")

    overlay_image = draw_move_overlay(
        st.session_state["base_image"],
        st.session_state["holds"],
        sequence,
        current_step
    )

    current_move = sequence[current_step]
    limb = current_move["limb"]
    hold = current_move["hold"]
    cue  = current_move.get("cue", "")

    hold_str = f"Hold {hold}" if hold is not None else "wall"
    st.markdown(f"""
    <div class="move-card">
        <div class="move-num">Move {current_step + 1} of {len(sequence)}</div>
        <div class="move-text">{html.escape(limb)} &rarr; {html.escape(hold_str)}</div>
        <div class="move-cue">{html.escape(cue)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.image(overlay_image, width=700)

    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("Previous", use_container_width=True):
            if st.session_state["current_step"] > 0:
                st.session_state["current_step"] -= 1
                st.rerun()
    with col_next:
        if st.button("Next", use_container_width=True):
            if st.session_state["current_step"] < len(sequence) - 1:
                st.session_state["current_step"] += 1
                st.rerun()

    # ---- Animated demo ----
    if "states" in st.session_state:
        st.markdown("""<p class="section-caption" style="margin-top:1.2rem;">Or watch the whole beta as an animation.</p>""", unsafe_allow_html=True)

        def _render_and_store_gif():
            with st.spinner("Rendering animation..."):
                gif_buf = render_beta_gif(
                    st.session_state["base_image"],
                    st.session_state["holds"],
                    sequence,
                    st.session_state["states"],
                    height_cm=height_cm,
                    wall_height_cm=wall_height_cm,
                )
                st.session_state["beta_gif"] = gif_buf.getvalue()
                save_route()

        if "beta_gif" not in st.session_state:
            if st.button("🎬 Animate the beta", use_container_width=True):
                _render_and_store_gif()
                st.rerun()

        if "beta_gif" in st.session_state:
            st.image(st.session_state["beta_gif"])
            col_re, col_dl = st.columns(2)
            with col_re:
                if st.button("Re-render animation", use_container_width=True):
                    _render_and_store_gif()
                    st.rerun()
            with col_dl:
                st.download_button(
                    "Download GIF",
                    st.session_state["beta_gif"],
                    file_name="climbai-beta.gif",
                    mime="image/gif",
                    use_container_width=True,
                )

# ---- Rate My Beta ----
if "annotated_image" in st.session_state and st.session_state.get("holds"):
    section_header("06", "Rate my beta", "Already have a sequence in mind? Describe it and get coaching feedback.")

    user_beta = st.text_area(
        "Describe your beta",
        placeholder="e.g. Start both hands on hold 5, LF on 2, RF on 1. Step RF up to 3, RH to 6, swap feet on 3...",
        height=150,
        key="user_beta_input",
        label_visibility="collapsed"
    )

    if st.button("Get coaching feedback", use_container_width=True, type="primary"):
        if not user_beta.strip():
            st.warning("Describe your sequence first — which hands and feet go where, in order.")
        else:
            try:
                guardrails.LEDGER.check("coach", VISITOR_ID)
                with st.spinner("Your coach is taking a look..."):
                    guardrails.LEDGER.count("coach", VISITOR_ID)
                    feedback = analyze_user_beta(
                        anthropic_client,
                        st.session_state["annotated_image"],
                        st.session_state["holds"],
                        user_beta,
                        height_cm, difficulty, wall_angle
                    )
                    st.session_state["beta_feedback"] = feedback
            except LimitExceeded as e:
                st.warning(str(e))
            except anthropic.APIError as e:
                log.warning("coaching API error: %s", e)
                st.error(friendly_api_error(e))
            except Exception:
                log.exception("coaching failed")
                st.error("The coach couldn't answer just now — try again in a moment.")

    if "beta_feedback" in st.session_state:
        st.markdown(st.session_state["beta_feedback"])

# ---- Save & export ----
if "sequence" in st.session_state and st.session_state.get("holds"):
    section_header("07", "Save this route",
                   "Keep it in your library for this browser — come back to this same link to find it again.")

    route_name = st.text_input(
        "Route name",
        placeholder="e.g. movement-blue-v2",
        key="route_name",
        label_visibility="collapsed"
    )

    col_save, col_export = st.columns(2) if DEV_MODE else (st.container(), None)
    with col_save:
        if st.button("Save to library", use_container_width=True):
            slug = _slugify(route_name)
            if not slug or slug == "last":
                st.warning("Give the route a name first.")
            else:
                save_route(os.path.join(VISITOR_DIR, slug))
                st.success(f"Saved as '{slug}'. It stays available from this link until the server restarts.")
    if DEV_MODE:
        with col_export:
            if st.button("Export as test route", use_container_width=True):
                slug = _slugify(route_name)
                if not slug:
                    st.warning("Give the route a name first.")
                else:
                    path = export_test_route(slug, colour, wall_angle, difficulty)
                    st.success(f"Exported to {path} — open route.md and fill in your verified beta after climbing it.")

if DEV_MODE:
    with st.expander("Developer: today's usage"):
        st.json(guardrails.LEDGER.status())

st.markdown("""<p class="climb-footer">climbai &mdash; get through the plateau</p>""", unsafe_allow_html=True)
