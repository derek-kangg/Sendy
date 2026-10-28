# Overnight notes — 2026-10-28

Read this first; the full audit is in [production-readiness.md](production-readiness.md).

## Where things stand

- Everything from last night is on the branch **`claude/production-hardening`**
  (this worktree), based on `main` at `1d620e5`. Nothing was pushed, nothing
  touched `main`, no deploy was changed.
- The old `claude/competent-mendel-542f29` branch was a pre-history-rewrite
  leftover with a stale `app.py`; its dirty edits are parked in commit
  `97d0ed8` on that branch and can be deleted.
- Run the tests (no API spend): `python -m unittest discover -s tests -t .`
- Run the app from this worktree with your existing venv:
  `C:\Users\derek\climbai\venv\Scripts\streamlit run app.py`
  (a copy of your `.env` is in the worktree, gitignored).

## What the audit concluded

Three judges (owner / senior engineer / climber-on-a-phone) scored three
deployment paths. **"Harden the existing Streamlit app and move it to
Railway" won clearly (≈7.5/10)** over a spend-fenced variant of the same
(6.7) and a FastAPI + PWA + Supabase rebuild (≈4.8). The rebuild is the
right *eventual* architecture, but 45-55 hours before anything usable
ships is where this project has stalled before.

Per-route cost today: **~$0.16 clean, ~$0.22-0.25 realistic, $0.86 worst
case** (Generate is 75% of it). With the prompt caching now on the branch
the typical route drops to roughly **$0.09**.

## What is done on the branch (all verified by tests or in the browser)

Engine / detection (headless, survive any future frontend):
- Phone photos normalised once at upload (EXIF orientation, 1600 px cap,
  bomb guard) — fixes the sideways-markers blocker and most of the RAM.
- Detection: typed errors instead of blank screens, 60-candidate cap,
  token budget + truncation check, timeout.
- Generation: strict validation (no hands-to-wall, feet above hands, or
  unvalidated third attempts), LH/RH by position, sit-start cues, explicit
  finish hold, wall height parameter, cached static prompt (~60% of
  Generate cost), per-call timeouts, and a `GenerationResult` with
  complete / incomplete / stalled / error so partial betas are honest.
- `guardrails.py`: kill switch, per-visitor and global daily caps, daily
  budget, metered client that prices every call. All env-tunable.

App:
- Per-visitor storage keyed by a random id in the URL (`?v=`); nobody can
  list anyone else's photos; the same link survives a refresh.
- **Generation runs in a background worker** and the page polls it, so
  locking the phone or a dropped connection no longer kills it; the job
  id is in the URL (`?job=`) so a reload reattaches.
- Tap image is a ≤900 px JPEG (**122 KB** measured, vs ~25 MB before).
- Stale-tap bug on mode switch fixed; start holds editable after
  generating; two-start-hold cap; selected start holds ringed in green.
- Wall-height input (replaces the dead "Start marking" control) and a
  finish-hold selector.
- Optional camera capture, privacy line, typed error messages everywhere,
  tracebacks hidden in production, detection cache per photo+colour,
  developer-only export behind `SENDY_DEV=1`, optional
  `SENDY_ACCESS_CODE` for a private beta.
- Deploy files: `Dockerfile`, `fly.toml`, `.dockerignore`,
  `.python-version`. Dead scratch files removed from the repo root.

## Smoke test (real API, 2026-10-28 06:30)

On the route-01 test wall, restored from a per-visitor link: tapped Hold 5
as the start, hit Generate, reloaded the page mid-generation (it
reattached and kept going), and animated the result.

- Beta: start both hands 5 / LF 2 / RF 1 (same start you climb it with),
  then RH→6, LH→7, RF→3, RH→8, LF→6, RH→9, match. Status `complete`.
- 8 model calls, **$0.042 total**; calls 2-8 cost $0.0045 each because the
  cached system block was hitting (the first call, which wrote the cache,
  was $0.010).
- Tap image shipped to the browser: 122 KB JPEG.

## Decisions only you can make (defaults recommended)

1. **Host** — recommended: Railway Hobby, one replica, app sleeping off.
   ~$5-7/mo fixed + Anthropic usage. Railway paused sign-ups on 2026-10-28, so Fly.io first (docs/deploy.md). Cloud Run (~$0 with a cold-start
   spinner) is the fallback if $5 is too much. Streamlit Community Cloud
   cannot be the product host (12 h sleep + manual wake).
2. **Monthly ceiling** — recommended: `SENDY_DAILY_BUDGET_USD=5`
   (≈$150/mo absolute worst case, realistically $10-20) *and* a hard
   spend limit on a dedicated Anthropic Console workspace.
3. **Private or public first** — recommended: private beta with
   `SENDY_ACCESS_CODE` for the gym friends for two weeks, then open.
4. **Web vs native** — recommended: this web app now; the FastAPI + PWA
   path later, once real climbers have used it and you have a measured
   per-route cost from the ledger.

## Your to-do before/at deploy (things I cannot do for you)

- [ ] **Rotate the Anthropic key.** The current key sat in Community
      Cloud secrets and in `.env`; create a dedicated workspace with a
      monthly spend limit and use that key on Railway.
- [ ] **Delete (or make private and unlink) the Community Cloud app** —
      `projectclimb.streamlit.app` is a sign-in wall today, and the README
      still links to it.
- [ ] Create the Railway project from this repo (after merge), set
      `ANTHROPIC_API_KEY` (+ optional `SENDY_ACCESS_CODE`), set replicas
      to 1, App Sleeping off, and a hard usage limit on the plan.
- [ ] Soak test from two phones: generate on both, lock one screen at
      30 s, pull-to-refresh the other, upload a non-image renamed `.jpg`.
- [ ] Review and merge `claude/production-hardening` into `main`.

## Known gaps left on purpose

- Detection colour ranges (red hue wrap, pink/black, white) and the tape
  filter are untouched — the manual editor is still the safety net.
- The library is per visitor and ephemeral (lost on redeploy); a durable
  per-user library needs accounts and storage (the PWA phase).
- The in-process ledger and job registry assume one replica.

## The App Store path (the end goal, kept in view)

Derek's stated destination is a native app in the App Store / Play Store.
The audit's Option 3 is that architecture; the judges ranked it last only
on *sequencing* — it is 45-55 hours of work before anything is usable,
which is where this project has stalled before — not on merit. The web
app ships first so real climbers are using Sendy while the native app is
built, and so the per-route cost is measured, not estimated.

What last night's work already contributes to the native app:

- `hold_detection.py`, `sequence_engine.py`, `route_graph.py`,
  `beta_animation.py`, `image_ingest.py` and `guardrails.py` are headless.
  They become the backend's brain unchanged; only `app.py` (Streamlit) is
  throwaway.
- `GenerationResult` + the progress callback are exactly what a `POST /beta`
  job endpoint streams to a phone.
- The ledger already prices every call, so the subscription/credit model
  can be designed from real numbers.

What it still takes (honest, solo, evenings + weekends):

1. **Backend** (~1-2 weekends): FastAPI on the same Railway box —
   `POST /detect`, `POST /beta` (job + progress via SSE), `POST /animate`,
   `POST /coach`; API key stays server-side; per-user quotas replace the
   per-visitor ones.
2. **Accounts + storage** (~1 weekend): Supabase auth + per-user routes and
   photos (Free tier to start; it pauses after a week idle, Pro is $25/mo).
3. **Client** (~2-3 weekends to a TestFlight beta): Expo / React Native —
   native camera, tap-to-select overlay drawn client-side, background
   generation with push/poll, GIF or in-app animation.
4. **Store overhead**: Apple Developer Program $99/yr, Google Play $25
   one-time, privacy policy page, app review (photo upload + AI output
   both need clear disclosure), screenshots.

Suggested gate before starting (1): two weeks of gym friends using the web
beta, at least ~30 real routes through the ledger, and the colour-detection
session's improvements merged. At that point the backend and client can be
planned as their own tracked piece of work.
