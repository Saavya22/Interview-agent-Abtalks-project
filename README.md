# AI Interview Agent

A conversational technical interviewer that runs a personalized, multi-turn
interview based on a candidate's actual progress through the 31-day AI Cohort
curriculum, and produces structured feedback at the end — with a formal,
transcript-styled web interface built in.

## What you get

- A backend (`main.py`) that matches the Technical Spec's `POST /api/interview`
  contract exactly.
- A frontend (`frontend/index.html`) served at `GET /` by the same app — so
  opening the deployed URL in a browser *is* the app. No separate hosting step.

## How it works

- **Single endpoint, per spec:** `POST /api/interview`, state keyed by
  `sessionId`, no auth, no persistent storage — matches `technical-spec.md`
  exactly (start with `candidate`, turn with `message`, end returns
  `feedback`).
- **Claude drives the conversation**, but every turn is forced through a tool
  call (`interview_turn`) so the model can *only* respond with a structured
  object: `{ reply, question_day, done, feedback }`. This removes all
  free-text parsing risk and gives the server a reliable signal for exactly
  which curriculum day each question targets.
- **Server-side guardrails, not just prompting.** The backend counts real
  questions asked and distinct curriculum days covered from `question_day`
  on every turn. The model is *not allowed* to end the interview
  (`done: true`) until it has asked at least 8 questions across at least 4
  distinct days — if it tries to wrap up early, the server transparently
  rejects that turn, nudges the model with the missing counts, and retries
  (up to 2x) before the candidate ever sees a reply. This guarantees the
  minimum requirements are met by construction, not by hoping the LLM
  follows instructions.
- **Personalized context, not a fixed question bank.** For each candidate,
  the system prompt includes only the curriculum days they actually
  attempted (with objectives + tools), their pass/fail/attempt history, and
  engagement signals. The model is instructed to:
  - go deeper on days with multiple attempts (likely struggle areas),
  - lightly check in on *skipped* days without penalizing hard for them,
  - give one solid question to days passed cleanly on the first try,
  - mix conceptual / applied-design / trade-off questions,
  - and generate follow-ups grounded in what the candidate actually just
    said (the model sees full conversation history every turn).
- **Persistent sessions everywhere.** `session_store.py` auto-switches
  between an in-memory dict (local dev) and Redis over REST (Vercel KV /
  Upstash) based on which env vars are present — `main.py` doesn't need to
  know which backend is active.

## The frontend

`frontend/index.html` is a single self-contained file (no build step) styled
as a formal interview transcript / dossier rather than a typical chat app:

- A letterhead with a brass seal and a monospace session ID, like a case
  file header.
- Turns rendered as a timestamped, speaker-tagged transcript (Interviewer /
  Candidate) instead of avatar chat bubbles — interviewer questions pick up
  a small day chip (e.g. `D07`) when the reply mentions a curriculum day.
- On completion, the transcript closes with a "Record Closed" rule and a
  parchment **Evaluation Report** card renders the structured feedback
  (summary, strengths, gaps, next steps) as a formal document.
- An intake screen lets you pick a candidate from the roster (fetched from
  `/api/candidates`) or paste a custom candidate JSON object, plus a
  collapsible "Connection settings" field if you ever host the frontend
  separately from the backend (defaults to same-origin, which is the setup
  used when `main.py` serves it directly).

## Run it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000/** — that's the interview UI itself. (The
interactive API docs are still available at `/docs` if you want to hit the
endpoint directly.)

## Deploying to Vercel

Sessions must survive across requests, but Vercel's Python functions are
stateless serverless instances — so `session_store.py` auto-switches from an
in-memory dict (local dev) to Redis over REST (Vercel KV or Upstash) the
moment it sees the right env vars. Nothing else changes.

1. Push this repo to GitHub (see file list below).
2. In the Vercel dashboard: **Add New Project** → import the repo. Vercel
   detects `vercel.json` and `api/index.py` automatically as a Python
   serverless function.
3. Add a KV store: **Storage** tab → **Create Database** → **KV** (or add the
   Upstash integration from the marketplace) → connect it to the project.
   This auto-populates `KV_REST_API_URL` / `KV_REST_API_TOKEN` (or the
   `UPSTASH_REDIS_REST_*` equivalents) as environment variables.
4. Add `ANTHROPIC_API_KEY` under **Settings → Environment Variables**.
5. Deploy. Your endpoint is `https://<project>.vercel.app/api/interview`.

Without a KV store attached, the app still deploys and works for a *single*
request-response pair, but multi-turn interviews will break unpredictably
whenever Vercel routes a request to a different instance — so step 3 isn't
optional for real use, only for a quick sanity check of step 2.

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app — backend + serves the frontend at `/` |
| `frontend/index.html` | The interview transcript UI (self-contained, no build step) |
| `session_store.py` | Session persistence: in-memory locally, Redis on serverless |
| `api/index.py` | Vercel entrypoint (re-exports the FastAPI app) |
| `vercel.json` | Vercel build/routing config |
| `curriculum.json` | 31-day curriculum reference (provided) |
| `sample_candidates.json` | Sample candidate profiles — powers the roster selector and local testing (provided) |
| `test_flow.py` | Offline test that mocks the Anthropic client to validate session state, the coverage guardrail/retry logic, and error handling — no API key or network needed |
| `requirements.txt` | Python deps |
| `.env.example` | Template for local env vars |
| `.gitignore` | Keeps `.env`, caches, etc. out of the repo |

## Design notes / trade-offs

- **Model:** defaults to `claude-sonnet-4-6`, overridable via
  `INTERVIEW_MODEL` env var.
- **Forced tool-calling over free text:** slightly more tokens per call, but
  eliminates an entire class of "did the model actually say done" parsing
  bugs and gives exact, auditable coverage tracking (`questions_asked`,
  `days_covered`) for free.
- **Guardrail retries are invisible to the candidate:** if the model tries to
  wrap up early, the retry happens *inside* the same HTTP request — the
  candidate only ever sees the eventual valid question, never a "sorry,
  continuing" artifact.
- **Hard cap (`MAX_QUESTIONS_HARD_CAP = 16`):** safety valve so a
  misbehaving model can't loop forever; not currently force-enforced beyond
  minimums since 16 is generous for realistic interviews, but the hook is
  there (`_run_turn`) to tighten if needed.
- Not implemented (per "Out of Scope"): auth, persistent accounts, voice,
  long-term history across sessions.
