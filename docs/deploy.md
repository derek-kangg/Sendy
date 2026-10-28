# Deploying Sendy

Target: one always-on container, private beta behind an access code, hard
spend limits on both sides. The repo ships a `Dockerfile` that runs the app
with `$PORT`, so the same image works on Fly.io, Cloud Run or Railway.

Hosting status, 28 Oct 2026: **Railway has paused new sign-ups** (recovering
from an incident since 21 Oct, no reopening date) — so the plan is **Fly.io**
now, which the hosting research ranked a close second: no sleep, seconds-not-
minutes restarts, a real volume, ~$6-7/mo for a 1 GB Machine. Cloud Run is
the ~$0 alternative if that's too much (section C). When Railway reopens,
section D applies and the Dockerfile is reused unchanged.

## 0. Before you start (10 minutes, both are security fixes)

1. **New Anthropic key with a spend limit.** platform.claude.com → Settings →
   Workspaces → *Create workspace* "Sendy" → Limits → set a **monthly spend
   limit** (e.g. $20) → API keys → *Create key*: scope it to the Sendy
   workspace, expiration **90 days (custom)**, linked to you. Copy it once.
   Then disable the old key and update your local `.env`.
2. **Retire the Community Cloud app.** share.streamlit.io → the
   `projectclimb` app → ⋮ → *Delete*.

## A. Fly.io (current plan)

### A1. One-time setup

```powershell
winget install --id Fly.Flyctl -e
```

(or `pwsh -Command "iwr https://fly.io/install.ps1 -useb | iex"`). Then:

```powershell
fly auth signup
```

Fly has no free tier: it asks for a card and gives a short trial. Set a
**spending limit** right away: fly.io dashboard → your organization →
Billing → *Spending limit* (e.g. $15/mo).

### A2. First deploy (from the repo root, on the merged branch)

```powershell
fly launch --no-deploy --copy-config --name sendy-<yourname> --region sea
```

`--copy-config` keeps the committed `fly.toml`; it only rewrites the `app`
line if the name is taken. Answer "no" to adding a Postgres/Redis. Then set
the secrets (never put the key in `fly.toml`):

```powershell
fly secrets set ANTHROPIC_API_KEY=sk-ant-... SENDY_ACCESS_CODE=<phrase> SENDY_DAILY_BUDGET_USD=5
```

and deploy:

```powershell
fly deploy
```

The first deploy builds the image on Fly's builders (3-5 minutes), creates
the 1 GB volume `sendy_data` in `sea`, waits for `/_stcore/health`, and
prints the URL (`https://sendy-<yourname>.fly.dev`).

### A3. Check it

```powershell
fly status
fly logs
```

`fly status` should show **1 Machine, started, health passing**. In the logs
every model call prints `claude call model=... cost=$0.00xx today=$0.0xx`.
Then the phone test: open the URL, enter the access code, upload a wall
photo, detect, tap a start hold, generate, **lock the phone for 30 s**,
unlock — the beta is there or still progressing; pull-to-refresh
mid-generation reattaches (`?job=` in the URL).

### A4. Operating it

| Want to… | Do |
|---|---|
| ship a change | `git push` then `fly deploy` (the volume keeps `/data`) |
| change a limit | `fly secrets set SENDY_DAILY_BUDGET_USD=10` (restarts the app) |
| emergency stop | `fly secrets set SENDY_DISABLED=1` |
| open it to everyone | `fly secrets unset SENDY_ACCESS_CODE` |
| see spend | Anthropic Console → Usage (Sendy workspace); Fly dashboard → Billing |
| scale memory if OOM | `fly scale memory 2048` **and** update `[[vm]]` in `fly.toml` |

Keep it at **one Machine**: the spend ledger and background generation jobs
live in the process. `fly.toml` already pins `min_machines_running = 1` and
`auto_stop_machines = "off"`.

## B. Build the image locally first (optional, recommended once)

With Docker Desktop running:

```powershell
docker build -t sendy .
docker run --rm -p 8080:8080 -e ANTHROPIC_API_KEY=sk-ant-... sendy
```

then open http://localhost:8080 and `curl http://localhost:8080/_stcore/health`
→ `ok`.

## C. Cloud Run (the ~$0 option)

Same Dockerfile. Trade-off: scale-to-zero means a 5-15 s spinner on the
first visit after idle, and WebSockets need `--session-affinity`,
`--timeout 3600`, `--max-instances 1`. Routes need a Cloud Storage FUSE
volume instead of a disk. Steps (after `gcloud init` and enabling Cloud Run
+ Artifact Registry):

```powershell
gcloud run deploy sendy --source . --region us-west1 --allow-unauthenticated `
  --memory 1Gi --cpu 1 --max-instances 1 --session-affinity --timeout 3600 `
  --set-env-vars SENDY_DAILY_BUDGET_USD=5 `
  --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest
```

Budget alert: Billing → Budgets & alerts. Expect ~$0/mo inside the free
tier at low traffic, ~$12/mo with `--min-instances 1` to kill the spinner.

## D. Railway (when sign-ups reopen)

New Project → Deploy from GitHub → it detects the Dockerfile. Variables as
in A2, plus `SENDY_DATA_DIR=/data` with a volume mounted at `/data`.
Settings → Deploy → Healthcheck Path `/_stcore/health`; replicas 1; App
Sleeping off; a hard usage limit on the plan. ~$5/mo on Hobby.
