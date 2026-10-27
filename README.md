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
pip install -r backend/requirements.txt          # serving only (ONNX runtime, no torch)
# or, for training / evaluation / tests:
pip install -r backend/requirements-train.txt
uvicorn backend.main:app --reload --port 8000
```

The API health endpoint is `http://localhost:8000/health` (it also reports which
classifier is loaded); the WebSocket endpoint is `ws://localhost:8000/ws`.
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
security incident), SOP gaps and the repeat-loop guard are hard overrides. Intents
below 0.5 confidence are reported as `uncertain` and ignored by routing; answers to a
slot question skip intent classification. For both, risk comes from emotion + emotion
keyword rules only, so a confident frustrated/angry message can still escalate alone. Groq only writes FAQ reply text, extracts
slot values and writes the session summary — it never picks a route.

Support requests go through slot filling (optional requester name/ID, category,
details, error, device/OS, impact), then the SOP self-help steps; a ticket is raised only if those don't fix it
or the employee asks. Ticket severity = f(peak risk, impact).

The runtime uses the best model present in `backend/ml/artifacts/`: `onnx/` (int8
ONNX export of the DistilBERT, served with onnxruntime — no torch), else
`multitask/` (PyTorch), else the TF-IDF baseline in `baseline/`; with none it runs
degraded (neutral prediction, keyword rules still active). Models load in a
background thread at startup — see "Deploy" below for downloading them from the
Hugging Face Hub.

```bash
python -m backend.ml.data.build_datasets   # Banking77+CLINC150+SNIPS+TWCS intents, GoEmotions emotions
python -m backend.ml.data.build_abcd       # ABCD per-turn escalation labels (no model yet)
python -m backend.ml.baselines             # keyword rules + TF-IDF/LogReg
python -m backend.ml.multitask --fp16      # multi-task DistilBERT (GPU / Colab)
python -m backend.ml.evaluate_ood          # 616 legacy IT messages, out-of-domain
python -m backend.ml.export_onnx           # int8 ONNX export + verification vs PyTorch
python -m backend.ml.upload_hub            # publish ONNX + TF-IDF to the HF Hub (needs HF_TOKEN)
```

Raw data is streamed into memory; nothing large is written to disk. Metrics land in
`backend/ml/results/`; weak-label samples for manual review in `backend/ml/data/review/`.

## Deploy the backend (Render free tier, 512 MB)

The deployed API serves the int8 ONNX model (66.8 MB) with onnxruntime — no torch.
Measured locally: ~11 ms per message on one CPU thread; ~324 MB resident while
serving, ~371 MB peak during startup (TF-IDF and ONNX briefly both loaded).

1. Publish the model (once per model version, from a training environment):
   ```bash
   python -m backend.ml.export_onnx      # writes backend/ml/artifacts/onnx/ + verification
   python -m backend.ml.upload_hub       # HF_TOKEN (write) in backend/.env; prints HF_MODEL_REVISION
   ```
2. Render web service:
   - Build command: `pip install -r backend/requirements.txt`
   - Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/health` (reports `classifier.model` / `classifier.state`)
3. Environment variables:

   | Variable | Value |
   |---|---|
   | `PYTHON_VERSION` | `3.11.0` (the TF-IDF fallback is pickled with scikit-learn 1.9.0 on 3.11) |
   | `GROQ_API_KEY` | your Groq key |
   | `DATABASE_URL` | the Render PostgreSQL connection string |
   | `FRONTEND_URL` | the deployed frontend origin, e.g. `https://closira.vercel.app` (CORS) |
   | `HF_MODEL_REPO` | `tanushreenerella/closira-multitask` |
   | `HF_MODEL_REVISION` | the commit SHA printed by `upload_hub` (pins the model version) |
   | `HF_TOKEN` | a **read** token — only needed while the Hub repo is private |
   | `CLOSIRA_ORT_THREADS` | optional, default `1` |

   Leave `CLOSIRA_CLASSIFIER` unset: it pins a model and disables background loading.

At startup the server answers immediately; a background thread downloads the TF-IDF
fallback (serves within seconds), then the ONNX model, which replaces it. Until the
first model is loaded the router runs on keyword rules only. Render's disk is
ephemeral, so the ~80 MB download repeats after each restart or spin-down.

## Limits

Responses are not token-streamed, one static SOP is used, and authentication is intentionally out of scope.
