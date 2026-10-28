# Closira: ML-Driven IT Helpdesk Assistant

An IT helpdesk assistant that decides **when an employee needs a human**. A multi-task DistilBERT reads each message for intent and emotion, and a calibrated risk router decides whether to keep helping or hand off with a ticket.

**[Live demo](https://closira-it-helpdesk.vercel.app/)** | **[Model on Hugging Face](https://huggingface.co/Tanushreenerella/closira-multitask)** | **[Detailed results](backend/ml/results/README.md)**

## Results

| | |
|---|---|
| Intent accuracy | **97.7%** (7 classes, baseline 34.1%) |
| Emotion accuracy | **80%** (6 classes, baseline 39.6%) |
| Escalation | **80% recall at 7.8% false alarms** (TF-IDF: 76% at 13.0%) |
| Served model | ONNX int8, **66.8 MB**, **11 ms** per message |
| Tests | 125 automated tests |

## How it works

```text
message -> DistilBERT (intent + emotion) -> risk router -> keep helping or hand off
```

- **The model scores, the router decides.** Routing is plain Python rules in `router.py`; the LLM (Groq) only extracts details and writes replies.
- **Slot filling:** asks only for missing details (issue, error, device, impact), one at a time.
- **SOP-grounded fixes:** self-help steps come only from `sop.json`; a ticket is created if they don't work.
- **Risk-based priority:** tickets are Critical, High, Medium or Low based on risk and impact.

## Datasets

| Dataset | Used for | Rows used |
|---|---|---|
| Banking77, CLINC150, SNIPS | Intent head | mapped to 7 support intents |
| Twitter Customer Support (TWCS) | Intent head (complaint, feedback, appreciation) | keyword-labelled tweets |
| GoEmotions | Emotion head | single-label comments, mapped to 6 emotions |
| 616 IT helpdesk messages | Escalation test only (never trained on) | 308 held out |

In total: **39,861 intent** and **31,979 emotion** training rows. Original labels were mapped to Closira's labels in `intent_map.json` and `emotion_map.json`.

**Intents:** information_request, issue_report, action_request, complaint, feedback, appreciation, chit_chat
**Emotions:** angry, frustrated, confused, sad, happy, neutral

## Model

- `distilbert-base-uncased` with **one shared encoder and two heads** (intent and emotion) on the [CLS] token.
- Trained with **alternating batches** (intent, then emotion), class-weighted cross-entropy, AdamW (lr 2e-5), early stopping. Best epoch 3 of 5, about 11 minutes on a Colab T4.
- One encoder instead of two models: about 66M parameters instead of 133M, and one forward pass per message.
- Exported to **int8 ONNX** (266 MB to 66.8 MB, 112 ms to 11 ms) with the same escalation results.

## Evaluation

- Compared against **keyword rules** and **TF-IDF + logistic regression** on the same splits.
- Escalation thresholds were tuned on one half of the 616 helpdesk messages and reported on the other half.
- **Limitations:** the 616 messages are LLM-generated, so real-world recall may be lower; most of the recall gain came from keyword rules, while the model mainly reduced false alarms. Next: a conversation-level model for repeated confusion across turns.

## Run locally

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000

cd frontend && npm install && npm run dev
```

Needs `GROQ_API_KEY` and `DATABASE_URL`. Training and evaluation commands are in [backend/ml/results/README.md](backend/ml/results/README.md).
