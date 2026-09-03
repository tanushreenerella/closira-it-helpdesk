# Closira — IT Helpdesk AI Support Assistant

LangGraph and Groq workflow for SOP-grounded employee IT support, issue qualification, escalation, and session summaries.

The assistant supports account and password issues, Wi-Fi and network connectivity, VPN access, managed-device problems, email issues, and approved software or access requests. It uses the SOP as its source of truth and routes sensitive, high-impact, or unsupported requests to Human IT Support.

## Project layout

```text
backend/
  main.py                FastAPI WebSocket API
  agent.py               LangGraph IT-support workflow
  cli.py                 terminal support runner
  sop.json               IT Helpdesk SOP and troubleshooting playbooks
  nlp/                   dataset, training scripts, runtime classifier, model artifact directory
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
python backend/cli.py
```

## Escalation classifier

`backend/nlp/classify.py` emits `frustrated`, `explicit_human_request`, `repeated_confusion`, or `normal`. Until `backend/nlp/bert_model/` contains a trained model, the agent preserves the prior keyword guards as its fallback.

```bash
python backend/nlp/generate_dataset.py
python backend/nlp/train_baseline.py
python backend/nlp/finetune_bert.py
```

## Limits

Responses are not token-streamed, one static SOP is used, and authentication is intentionally out of scope.
