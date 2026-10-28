# Deploying Sendy to Railway

Target: one always-on Railway Hobby service, private beta behind an access
code, hard spend limits on both sides. Everything below is clicking; the
repo already carries what the build needs (`requirements.txt` pinned,
`.python-version` = 3.12, `railpack.json` with the Streamlit start command,
`.dockerignore`).

Why not Streamlit Community Cloud: free apps sleep after 12 h and a visitor
has to click a wake button, and today the URL is a sign-in wall. Why Railway
over Cloud Run / Fly / Render: no sleep, no Dockerfile, one replica is the
default, ~$5/mo all in (see `production-readiness.md` §3 for the comparison).

## 0. Before you start (10 minutes, both are security fixes)

1. **New Anthropic key with a spend limit.** console.anthropic.com →
   Settings → Workspaces → *Create workspace* "Sendy" → in that workspace,
   Limits → set a **monthly spend limit** (e.g. $20) → API Keys → *Create
   key*. Copy it somewhere safe; you paste it into Railway in step 2.
   Then disable the old key (the one in `.env` / Community Cloud): Settings
   → API Keys → … → Disable. Update your local `.env` with the new key.
2. **Retire the Community Cloud app.** share.streamlit.io → the
   `projectclimb` app → ⋮ → *Delete*. (The README no longer links to it.)

## 1. Create the Railway service

1. railway.com → sign in with GitHub → *New Project* → **Deploy from GitHub
   repo** → `derek-kangg/ProjectClimb`. Pick the branch: `main` after you
   merge the PR, or `claude/production-hardening` to try it first.
2. Railway will start a build immediately. It will **fail or crash-loop
   until step 2 is done** (no API key yet) — that's expected.

## 2. Variables (service → *Variables* tab → *Raw Editor* is fastest)

```
ANTHROPIC_API_KEY=sk-ant-...          # the new Sendy-workspace key
SENDY_ACCESS_CODE=<pick a short phrase>   # private beta; remove later to open it up
SENDY_DAILY_BUDGET_USD=5
SENDY_MAX_GENERATIONS_PER_DAY=150
SENDY_MAX_GENERATIONS_PER_VISITOR=8
SENDY_DATA_DIR=/data                  # only once the volume in step 4 exists
```

Optional: `SENDY_DISABLED=1` is the kill switch (AI actions refuse,
existing routes still display). `SENDY_DEV=1` shows the usage ledger and
the test-route export — don't set it on the public deployment.

## 3. Settings tab

- **Networking → Generate Domain.** When asked for a port, enter `8080` or
  leave it to Railway; the start command binds to Railway's `$PORT`
  either way. This is the URL you share.
- **Deploy → Healthcheck Path:** `/_stcore/health` (Streamlit answers `ok`
  only when the Python app is actually up). Healthcheck timeout: 120.
- **Deploy → Restart Policy:** On Failure, 5 retries.
- **Deploy → Replicas:** 1 (default). Do not scale out: the spend ledger
  and the background generation jobs live in the one process.
- **App Sleeping:** leave **off** (default). Sleeping would bring back the
  "wake up" delay we are leaving Streamlit Cloud to escape.
- Start command: already provided by `railpack.json`. If the deploy log
  shows it running `python app.py` instead of `streamlit run …`, set
  *Deploy → Custom Start Command* to
  `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`.

## 4. Volume (so routes and access-code unlocks survive redeploys)

Right-click the service on the canvas → *Attach Volume* → mount path
`/data` → then add `SENDY_DATA_DIR=/data` to Variables. 1 GB is plenty
(a route is ~2-5 MB and visitor folders are purged after 24 h).

## 5. Plan limits (so a bug can't become a bill)

Workspace → *Usage* → set a **hard usage limit** (e.g. $10/mo) and an
email alert at $5. Together with the Anthropic workspace limit, the worst
month is bounded on both sides.

## 6. Verify

1. `https://<your-domain>/_stcore/health` → `ok`.
2. Open the app on your phone, enter the access code, upload a wall photo,
   detect, tap a start hold, generate, lock the phone for 30 s, unlock —
   the beta should be there or still progressing.
3. Pull-to-refresh mid-generation: it reattaches (`?job=` in the URL).
4. Railway → *Deploy Logs*: each model call logs
   `claude call model=... cost=$0.00xx today=$0.0xx`.
5. Hand the link + code to two friends and generate at the same time.

## Operating it

- **Change a limit** → edit the variable; Railway redeploys automatically.
- **Emergency stop** → `SENDY_DISABLED=1` (no redeploy of code needed).
- **Open it up** → delete `SENDY_ACCESS_CODE`.
- **Watch spend** → Anthropic Console → Usage (the workspace), and the
  Railway deploy log lines above. Expect ~$0.04-0.10 per route.
- Pushing to the deployed branch redeploys. The volume keeps `/data`; the
  in-process ledger resets on each deploy (that's fine — the Anthropic
  workspace limit is the hard backstop).
