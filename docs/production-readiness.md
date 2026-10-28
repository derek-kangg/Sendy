# ClimbAI: deployment readiness assessment

Scope: `C:\Users\derek\climbai` as it stands on `main` (the uncommitted `claude/production-hardening` worktree already covers some engine items; noted where it matters). Line references are to `app.py`, `hold_detection.py`, `sequence_engine.py`, `route_graph.py`, `beta_animation.py` on `main`.

## 1. Why it feels unusable

**The single biggest reason: the core phone flow is broken by three independent defects, any one of which stops a climber at step 3.** Nothing about the idea is wrong; the pipeline was built and tested on one laptop with one landscape-pixel photo, and real phone use never went through it.

### Phone photos break the pipeline before you ever see holds
- **25 MB tap image** (verified 3/3). `app.py:529` passes the full-resolution annotated image to `streamlit_image_coordinates` with the component defaults `image_format="PNG", png_compression_level=0`. Measured on your own saved 2160x2880 route: 18.7 MB PNG -> 24.9 MB base64 data URI, vs 0.43 MB as JPEG. A 12 MP phone photo is ~49 MB. On gym LTE the hold image takes tens of seconds to minutes to appear, and again after every Add/Remove hold. Mobile Safari may discard the tab outright.
- **EXIF rotation** (verified 3/3). `cv2.imread` at `hold_detection.py:134` honours the orientation tag; the four `Image.open` calls (`:251, :265, :337, :387`) do not. Portrait phone photos stored as landscape pixels + orientation=6/8 (standard on Android and many camera apps) get detected in one frame and drawn in another: sideways wall, numbers off the holds, taps matching the wrong coordinates. Your saved photo has no orientation tag, which is why you never saw it.
- **No resizing anywhere** (evidence-backed). Full-res is used for OpenCV, both Claude images, the annotation, the click component, and lives in `session_state` per session. ~120 MB peak per detect at 6 MP; a 12 MP upload doubles it on a 1 GB box. 48 MP phone modes (8064 px) are hard-rejected by the API and shown as a "network hiccup". No upload size cap (`.streamlit/config.toml` has no `[server]` section, so 200 MB default) and no pixel check before decode, so one oversized PNG OOMs the only process for everyone.

### Generate throws away your money and your work
- **Synchronous multi-minute loop killed by any rerun or disconnect** (verified 3/3). `app.py:614-662` runs up to 60 sequential Claude calls inside the button branch. `progress_bar.progress()` at `:633` is exactly where Streamlit injects `RerunException`/`StopException`; those are `BaseException`, so `except Exception` at `:647` never sees them. Screen lock, app switch, a second tap on Generate (no in-progress guard), or any widget tap aborts the run silently. The sequence is written to `session_state` only after the whole loop succeeds (`:657`), so every paid move is gone. A disconnect over 2 minutes also drops holds and start selection.
- **Progress bar divides by a fixed 20** (`:633`), so real 8-move routes stall at 40% and vanish. Climbers assume it hung and tap again (double spend).
- **Any API exception mid-run discards everything and says "network hiccup"** (`:647-650`), no logging anywhere.

### The beta is physically wrong on most real photos
- **Hard-coded 4 m wall** (evidence-backed blocker). `route_graph.py:9-10` on `main`: `WALL_HEIGHT_CM = 400; pixels_per_cm = image_height / WALL_HEIGHT_CM`, duplicated at `sequence_engine.py:99`. A normal shot from the mats frames 5-6 m (reaches suggested 25-50% too long, holds skipped); a tight crop frames 2 m (reachable moves rejected, engine burns corrections). No wall-height input exists. Your worktree already parametrises `wall_height_cm`; the UI does not expose it yet.
- **Third attempt accepted unvalidated** (`sequence_engine.py:318-324`): validates before each correction, never after the last. Probe: 20/20 invalid moves accepted over a 60-call run. The docstring's "every move passes physical validation" is false.
- **Foot and both-hands moves bypass reachability** (`:278-285`); foot list uses the hand radius (2 m high-steps accepted). **Sit starts** yield feet=None, which switches off the over-extension guard and prints "LF Hold None" as the first cue (`:110-120`).
- **Finish = highest detected blob** (`:93`); finish-style dropdown changes prose only. On your own saved route the highest hold is #19, a foot chip. **Two-hand start LH/RH assigned by tap order** (`:124`), so half of two-hand starts read crossed. **Never reaching the finish is silent**: 20-move cap, missing tool_use, invalid hold all return a partial beta the UI presents as complete.

### Safe public deployment is impossible as written
- **Shared on-disk library** (verified 3/3). `SAVE_DIR` is one global `saved_routes/last` (`app.py:34`). `save_route()` copies every visitor's photo after every detect/edit/generate/animate (`:497, :561, :662, :723`); `list_saved_routes` + the loader (`:501-512`) show every folder to any visitor who has not uploaded. Concurrent users clobber each other; strangers see each other's gym photos (often with other people in them). Writes are non-atomic. "Load it any time" is false on ephemeral disk.
- **Open wallet.** One global client with your key (`app.py:29`), no auth, no per-session counter, no daily budget. Worst-case Generate: 20 moves x 3 attempts = 60 calls, ~$0.80. `extra_notes` (`:472`) and `user_beta` (`:750`) have no `max_chars` and are injected verbatim into prompts (`sequence_engine.py:145`, `app.py:227`), so Rate My Beta is a general Claude endpoint anyone can drive.
- **Dead URL.** `README.md:6` 303s into a Streamlit sign-in wall, not even a wake page.

### Smaller things that compound the above (majors, verified)
- Zero-hold detection is a blank screen: `app.py:514` gates on truthy `holds`, so a wrong colour pick renders nothing, no hint, and still autosaves an empty route. On your photo `Red` yields 0 candidates; on the test wall `Orange` and `Yellow` do too.
- Stale tap replay on mode switch (`:551-553` dedups on (x,y), ignores the component's `unix_time`): switching the radio adds a phantom hold or deletes a real one and wipes the beta.
- Start holds silently locked once a beta exists (`:592`): the most common correction becomes impossible without re-detecting.
- `max_tokens=1500` on detection (`hold_detection.py:302`) truncates past ~40-50 candidates; `stop_reason` is never checked. Your photo produces 67 Purple / 59 Blue candidates. Result: a KeyError shown as "network hiccup" (deterministic, re-billed on retry) or unvalidated tail candidates kept as holds.
- Colour detection: Red lacks the 170-180 hue wrap; Pink/Black ranges miss real holds and catch grey ones; White segments the wall itself (0 of 3 real white holds found on the test wall); the tape filter deletes taped start holds and lets tape through as "footholds"; the validation schema can only reject `tape`, never other-colour holds, and cannot add missed ones.
- Animation crashes with a raw traceback when a hand is on `None` (`beta_animation.py:120`), which top-out finishes produce.
- Temp file written on every rerun, never deleted (`app.py:477-479`); "Export as test route" writes into the server's source tree and tells phone users to open a file they cannot reach.

## 2. What one route actually costs

Sonnet 4.6 at $3/$15 per M tokens, no thinking, no caching (the prompt puts volatile state first at `sequence_engine.py:211-215`, so nothing is cacheable today).

| Action | Calls | Typical | Worst |
|---|---|---|---|
| Detect holds | 1 (2 images, ~4,300 in) | $0.024 | $0.037 |
| Generate beta | ~13 (10 moves + 3 corrections, ~2,700 in each) | $0.12 | $0.80 (60 calls) |
| Rate My Beta | 1 | $0.017 | $0.034 |
| **Clean route** | | **~$0.16** | **~$0.86** |
| Realistic route (1.3 detects, 1.5 generates, 0.7 rates) | | **~$0.22-0.25** | ~$1.65 |

Generate is ~75% of the bill; ~75-80% of each Generate call's input is a static block re-sent every move.

| Routes/month | Clean typical | Realistic | All worst-case |
|---|---|---|---|
| 50 | ~$8 | ~$11 | ~$43 |
| 500 | ~$79 | ~$110 | ~$430 |
| 5,000 | ~$790 | ~$1,100 | ~$4,300 |

With prompt caching (static ~2,100-token prefix as a cached `system` block): Generate ~$0.12 -> ~$0.05, route ~$0.16 -> ~$0.09, 5,000 routes ~$790 -> ~$460.

What this means: a free public app with no gate is not a cost problem at friend scale (~$10/month) and an unbounded liability the moment the link escapes. One bored person mashing Generate on a busy wall is ~$0.80 per tap; a scripted loop is on the order of $1k/day. Image resolution is not a cost lever (the API downsizes to ~1.15 MP regardless); auth, per-session caps, a Console spend limit and caching are.

## 3. Deployment paths

**Option 1 — Harden the Streamlit app, move to Railway. Judges: 8.0 and 7.0 / 10.** Keep `app.py`, fix ingestion (EXIF + downscale + JPEG click image), delete the shared library instead of fixing it (JSON download/upload replaces it), run generation on a background thread polled by `st.fragment`, add an access code + per-session/daily counters + Console cap, and ship the small engine patches (wall height, LH/RH sort, validate-after-every-attempt, finish-hold). Railway Hobby $5/month, always-on, no Dockerfile. ~25-30 h; with your worktree branch already holding `image_ingest.py`, `guardrails.py` and the rewritten engine, closer to 15-20 h of app.py wiring and deploy. Leaves unfixed: Streamlit's session model (pull-to-refresh still blanks the app unless the job-registry graft below is added), colour detection quality, no real accounts, Streamlit UX ceiling.

**Option 2 — Fenced Streamlit on Railway with a volume. Judges: 6.7 / 10.** Same shape plus a dedicated Anthropic workspace, on-disk daily budget, `ThreadPoolExecutor(2)`, per-session storage keyed by `?s=` in the URL with `partial.json` resume, prompt caching, `html.escape` on the cue, typed error chain. The most correct engineering, and the only plan that survives an iOS tab discard mid-generation. But 35-45 h, first public deploy on day 6-7, three new subsystems (volume + sweep, `budget.py`, bearer-token URLs that expose the photo to anyone with the link), and it explicitly ships the 4 m wall scale, locked start holds, and crossed starts unfixed: "safe and durable, not the beta more right". Wrong trade for a product whose value is the beta.

**Option 3 — FastAPI + React PWA + Supabase. Judges: 5.0 and 4.5 / 10.** Right destination: the engine is already headless (`sequence_engine.py:1-8`, `scripts/eval_route.py:90-95` drives it without Streamlit), and a PWA structurally removes the rerun bugs, the 25 MB data URI, the shared folder, and gets native camera capture and real per-user storage. But 80-110 h, nothing public for 4-5 weekends, 35-45 h in a stack you have not shipped in (React/TS, JWT, CORS, iOS PWA quirks), three providers to babysit, and durable storage of gym photos that makes delete/retention mandatory. Its own first risk is "two months with nothing deployed". The first ~15 h of engine work are identical to Option 1 and are already mostly on your branch.

## 4. Recommendation

**Option 1, with the Option 2/3 grafts below. Ship weekend 1 even if weekend 2 slips.** Treat Option 3 as the phase after friends have used this for a month and you have a measured per-route cost.

Decisions you have to make, with defaults:
- **Budget per month:** default **$30 hard ceiling** = Railway $5 + Anthropic Console workspace limit $25. Railway hard usage limit $10 on top. Raise only after a month of real numbers from the usage log.
- **Web vs native:** default **web (Streamlit now, PWA later)**. No native app until the PWA has repeat users; the API layer serves both unchanged.
- **Library:** default **none on the server**. Route JSON on the climber's phone. Real accounts come with Option 3.
- **Access:** default **shared access code** written on the gym whiteboard, rotated by env var. It is friction, not auth; the counters and Console cap are the backstops.

### Stage 1 — first two weekends: safe, fast, always-on

Before any code (Saturday morning, ~2 h):
- Create a dedicated Anthropic Console workspace with a new key and a $25 monthly limit. Rotate and delete the old key (it was in Community Cloud secrets and sits in `.env`).
- Delete the Community Cloud app outright; fix `README.md:6`.
- Railway project from GitHub: `Procfile` (`web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`), `.python-version` = 3.12, env vars `ANTHROPIC_API_KEY`, `CLIMBAI_ACCESS_CODE`, one replica, App Sleeping off, health check `/_stcore/health`, hard usage limit $10. Add `.railwayignore` for `venv/`, `saved_routes/`, `test_routes/`, `yolov8n.pt`, `wall*.jpg`, `*_test.py`, the stray root scripts, and delete the `0:` artifact in the worktree.
- Finish the two failing worktree tests first (`test_hand_to_wall_is_rejected_then_corrected`, `test_over_extension_requires_a_foot_move_first`); run `scripts/eval_route.py` on route-01 and keep the scores as the regression gate for everything below.

Code, weekend 1:
- **Ingest once per `uploaded_file.file_id`** replacing `app.py:476-479`: reject `w*h > 24e6` with a clear message, `ImageOps.exif_transpose`, RGB, `thumbnail((1600,1600))`, JPEG q85 to one temp path, unlink the previous, pop holds/annotated/sequence/states/start_holds/beta_gif/last_click so a new photo cannot inherit the old route. Wire `image_ingest.py` from the branch. Offer `st.camera_input` beside the uploader feeding the same path.
- `app.py:529`: add `image_format="JPEG", jpeg_quality=80`. Use the same downscaled base for `draw_move_overlay`/`st.image`.
- `.streamlit/config.toml`: `[server] maxUploadSize = 10`, `[client] showErrorDetails = "type"`.
- **Delete persistence**: remove the four `save_route()` calls (`:497, :561, :662, :723`) and the Load-route branch (`:501-512`); gate section 07 behind `CLIMBAI_DEV`. Add "Save route to my phone" `st.download_button` (JSON with embedded JPEG) and an "Open a saved route" JSON uploader. One line of copy: "photos are processed in memory and not kept."
- **Spend gate**: access-code `text_input` gating everything; per-session counters (3 detects / 3 generates / 5 coachings); `@st.cache_resource` daily counter with a `Lock`; `max_chars=300` on `extra_notes` (`:472`), `800` on `user_beta` (`:750`); `max_calls=30` per Generate. Build the client as `anthropic.Anthropic(api_key=..., timeout=60.0, max_retries=2)` at `:29`. Reuse `guardrails.py`'s metered client and log `response.usage` per call with the session id.
- **Honest failures**: one most-specific-first chain (`AuthenticationError`, 402/billing -> "out of budget, try later", `RateLimitError`, `BadRequestError`, `APITimeoutError`/`APIConnectionError`) with `logging.exception` and request id at `:498, :647, :771`; reuse the engine's `_friendly_api_error()` for detect and coach. `st.warning("No <colour> holds found ...")` at `:514`, no autosave on empty. Try/except around `_render_and_store_gif` (`:713`). `html.escape(cue)` before `:691`.
- `hold_detection.py:302`: `max_tokens=4096`, check `stop_reason == "max_tokens"`, cap candidates at the 60 largest (the branch already has this).
- Per-session detection cache keyed by `sha256(jpeg bytes) + colour` so a second tap is free.
- Push, test from a phone on LTE with a portrait Orientation=6 photo and a huge PNG. **This is the first real deploy.** Weekday evening: three real routes at the gym, write down every stall.

Code, weekend 2:
- **Background generation**: module-level `ThreadPoolExecutor(max_workers=2)` in `@st.cache_resource`; job object (bytes, holds, start/finish, moves so far, `GenerationResult`) in a process-wide registry keyed by a uuid written to `st.query_params`, expiring after ~1 h, so pull-to-refresh or an iOS tab discard re-attaches instead of orphaning a thread that keeps spending. Engine gets `on_move(move, state)` and `resume_from=(sequence, states)`; the worker never touches `st.*`. `@st.fragment(run_every=2)` shows accepted moves, elapsed time, "about a minute, keep the screen on"; `generating` flag disables the button; third concurrent climber sees "queued". Resume is mandatory, not polish.
- **Engine wiring** (branch already has the engine side): "Wall height in this photo (m)" `number_input` default 4.0 with hint "mats to top of panel" -> `wall_height_cm`; "Select finish hold" click mode, default highest hold whose `best_use` is not foothold, "Start: 5 - Finish: 14" above Generate, reject start == finish; LH/RH sorted by x; validate-after-every-attempt with `reached_finish`/`stop_reason` surfaced as a warning + retry; allow start re-selection after a beta (clear sequence with a notice).
- **Tap UX**: dedup on `value["unix_time"]`, store it unconditionally before branching, pop on detect/new upload; draw selected start/finish rings via `redraw_annotation(..., start_nums, finish_num)` and `st.rerun()`; toast on a miss; "Click" -> "Tap".
- Sunday: two-phone soak at the gym (generate simultaneously, lock one at 30 s, pull-to-refresh the other, upload a renamed non-image and an oversized PNG, exhaust the daily cap and read the copy, check Railway memory graph). Update README (no library claim, document access code and JSON flow). Tag v0.1. Hand the code to three or four friends; watch logs and the usage page for a week.

### Stage 2 — weeks 3-6: cheaper and more right
- **Prompt caching**: static blocks (mechanics, foot economy, technique library, hold table, finish/wall/climber) into `system=[...cache_control ephemeral...]`, volatile state/history/reachable lists in the user turn; verify `cache_read_input_tokens > 0` on move 2; eval before/after. ~1 h, cuts Generate input ~60%. The branch already has a `cache_control` system block; do not regress it.
- **Validator**: foot graph with a foot radius (~0.6 x height) and both-hands against the hand graph; sit-start feet seeded on the start hold / wall with the over-extension guard falling back to the lowest hand; LF None guard in the cue. Eval-gated.
- **Detection**: Red as two hue ranges, lower Pink floors, Black relative to wall median V; tape test per watershed piece; extend the validation enum with `not_a_hold`/`other_colour` + `missed_holds`; treat unlabelled candidates as unvalidated. Then the real fix: tap one hold to sample the colour window instead of the dropdown.
- Fix the GIF font glyphs, hand-on-wall keypose, dead `start_style` selectbox, stale coaching across routes.
- Decide from the usage log whether 3 generates/session is too tight and whether the $25 cap needs raising.

### Stage 3 — when friends use it twice a week: Option 3
- FastAPI over the unchanged engine (`/routes`, `/beta` as a job with per-move persistence, `/animate`, `/coach`), Supabase auth + per-user private bucket with delete-route/delete-account, React PWA with `capture="environment"` and SVG overlays, Cloudflare Pages. Keep `app.py` as the local dev harness. Native wrapper only after the PWA has repeat users.

## 5. Refuted / not actually problems

- **Temp file per rerun as a MAJOR** — refuted 2/3: the leak is real (`app.py:476-479`, nothing ever unlinks) but Community Cloud disk is 50 GB and wiped on sleep, so it never bites at this traffic; downgraded to minor, and Stage 1's write-once-per-`file_id` fixes it anyway.
- **Path traversal via route name** — `_slugify` reduces to `[a-z0-9-]`, so `os.path.join` stays under the roots; only collisions and `'last'` overwrite remain (covered by removing the library).
- **"Load route" selectbox injection** — Streamlit sends an index, not a path.
- **Secrets in git** — `.env` and `secrets.toml` never committed; `git grep sk-ant-` across all revisions is empty; the key never reaches the browser. (Rotate it anyway: it was live on a public deploy.)
- **Malformed `same_hold_as` from the model** — `_resolve_merge_groups` guards out-of-range and cyclic values.
- **Tap-coordinate rescale** at `app.py:535-538` is correct.
- **GIF size/render time** — 0.2-0.6 MB and 1.5-3 s even on a simulated 12 MP photo; not a problem.
- **Image resolution as a cost driver** — the API downsizes to ~1.15 MP regardless; downscaling is a latency/RAM fix, not a token fix.
- **SDK retries as a cost multiplier** — failed 429/5xx attempts are not billed; the manual "hit generate again" path is the multiplier.
- **Watershed splitter and climbing_knowledge.py** — behaved sensibly in every trace; no factually wrong technique statements (one phrasing nit: "expect to cut smaller footholds" should read "expect smaller footholds").
- **Columns on phones / dark palette** — `st.columns(2)` stacks on narrow viewports; config sets `base="dark"` so the hard-coded CSS is consistent.