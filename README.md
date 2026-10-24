# Closira — IT Helpdesk AI Support Assistant

Plain-Python pipeline (local ML classifier + deterministic router + Groq reply text) for SOP-grounded employee IT support, issue qualification, escalation, and session summaries.

The assistant supports account and password issues, Wi-Fi and network connectivity, VPN access, managed-device problems, email issues, and approved software or access requests. It uses the SOP as its source of truth and routes sensitive, high-impact, or unsupported requests to Human IT Support.

## Project layout

```text
backend/
  main.py                FastAPI WebSocket API
  pipeline.py            run_turn(state, msg): classify -> route -> handler
  router.py              deterministic routing, risk score, keyword hard overrides
  cli.py                 terminal support runner
  sop.json               IT Helpdesk SOP and troubleshooting playbooks
  ml/                    intent + emotion models: data builds, baselines, multi-task DistilBERT, results/
  legacy/nlp/            old 4-label DistilBERT escalation classifier (kept for reference)
  database.py             PostgreSQL session persistence
frontend/
  app/                   Next.js App Router UI
  components/            reserved for reusable UI components
```

## Prerequisites

- Python 3.10+, a `GROQ_API_KEY`, and PostgreSQL
- Node.js 18.17+ and npm

Set the API key in your shell (or in the repository `.env`):

```bash
export GROQ_API_KEY=gsk_...
export DATABASE_URL=postgresql://closira_user:password@localhost:5432/closira
```

On Windows:

```cmd
set GROQ_API_KEY=gsk_...
set DATABASE_URL=postgresql://closira_user:password@localhost:5432/closira
```

## Run the backend

From the repository root:

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

The API health endpoint is `http://localhost:8000/health`; the WebSocket endpoint is `ws://localhost:8000/ws`.
The backend creates the required PostgreSQL tables at startup.

## Run the frontend

In another terminal:

```bash
cd frontend
npm install
copy .env.local.example .env.local
npm run dev
```

On macOS/Linux use `cp` instead of `copy`. `NEXT_PUBLIC_WS_URL` defaults to `ws://localhost:8000/ws`; override it in `.env.local` when the API is elsewhere. Open `http://localhost:3000`.

## CLI

```bash
python -m backend.cli
```

## Routing and the ML classifier

Each turn runs `classify -> route -> handler` (`backend/pipeline.py`). The classifier
predicts one of 7 intents (`information_request`, `issue_report`, `action_request`,
`complaint`, `feedback`, `appreciation`, `chit_chat`) and one of 6 emotions
(`angry`, `frustrated`, `confused`, `sad`, `happy`, `neutral`); `router.risk_score`
turns them into a per-turn risk in [0, 1]. `router.route` picks `faq | qualify |
escalate | summary`. Keyword rules (anger, explicit human request, repeated failure,
security incident), SOP gaps and the repeat-loop guard are hard overrides. Groq only
writes FAQ reply text and the session summary — it never picks a route.

The runtime loads `backend/ml/artifacts/multitask/` if present, else the TF-IDF
baseline in `backend/ml/artifacts/baseline/`, else runs degraded (neutral
prediction, keyword rules still active).

```bash
python -m backend.ml.data.build_datasets   # Banking77+CLINC150+SNIPS+TWCS intents, GoEmotions emotions
python -m backend.ml.data.build_abcd       # ABCD per-turn escalation labels (no model yet)
python -m backend.ml.baselines             # keyword rules + TF-IDF/LogReg
python -m backend.ml.multitask --fp16      # multi-task DistilBERT (GPU / Colab)
python -m backend.ml.evaluate_ood          # 616 legacy IT messages, out-of-domain
```

Raw data is streamed into memory; nothing large is written to disk. Metrics land in
`backend/ml/results/`; weak-label samples for manual review in `backend/ml/data/review/`.

## Limits

Responses are not token-streamed, one static SOP is used, and authentication is intentionally out of scope.
