"""Upload the served model files to the Hugging Face Hub as one commit.

    python -m backend.ml.upload_hub [--repo Tanushreenerella/closira-multitask] [--public]

Uploads, in the layout backend/ml/inference.py downloads into artifacts/:
    onnx/model.int8.onnx, onnx/tokenizer.json, onnx/labels.json
    baseline/intent.joblib, baseline/emotion.joblib      (TF-IDF fallback)
    README.md                                            (model card)

HF_TOKEN (a write token) is read from the environment / backend/.env / .env.
The repo is created private unless --public. Prints the commit SHA to set as
HF_MODEL_REVISION on the server.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ML_DIR = Path(__file__).resolve().parent
ROOT = ML_DIR.parents[1]
ARTIFACTS_DIR = ML_DIR / "artifacts"
RESULTS_DIR = ML_DIR / "results"
FILES = ["onnx/model.int8.onnx", "onnx/tokenizer.json", "onnx/labels.json",
         "baseline/intent.joblib", "baseline/emotion.joblib"]


def model_card(repo: str) -> str:
    def load(name):
        path = RESULTS_DIR / name
        return json.loads(path.read_text()) if path.is_file() else {}
    onnx, acc = load("onnx_export.json"), load("accuracy.json")
    labels = json.loads((ARTIFACTS_DIR / "onnx" / "labels.json").read_text())
    lines = [
        f"# {repo}", "",
        "Multi-task DistilBERT (`distilbert-base-uncased`) for the Closira IT helpdesk router: one encoder, an intent "
        "head and an emotion head, exported to ONNX with dynamic int8 quantization. Served with onnxruntime + "
        "tokenizers (no torch). `baseline/` holds the TF-IDF + LogReg fallback.", "",
        f"- intents: {', '.join(labels['intent'])}",
        f"- emotions: {', '.join(labels['emotion'])}",
        "- inputs: `input_ids`, `attention_mask` (int64, max length "
        f"{labels.get('max_length', 128)}); outputs: `intent_logits`, `emotion_logits`", "",
    ]
    if acc:
        lines += ["| Eval (PyTorch model) | Accuracy | Macro-F1 |", "|---|---|---|",
                  f"| intent test | {acc['intent_test_split']['accuracy']} | {acc['intent_test_split']['macro_f1']} |",
                  f"| emotion test | {acc['emotion_test_split']['accuracy']} | {acc['emotion_test_split']['macro_f1']} |", ""]
    if onnx:
        agree = {t: v["prediction_agreement"] for t, v in onnx["tasks"].items()}
        lines += [f"int8 vs PyTorch prediction agreement: intent {agree['intent']}, emotion {agree['emotion']}.", ""]
    lines += ["Training data: Banking77, CLINC150, SNIPS, TWCS (keyword weak labels), GoEmotions. "
              "See backend/ml/results/README.md in the Closira repo for evaluation details and caveats."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="Tanushreenerella/closira-multitask")
    p.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    args = p.parse_args(argv)

    load_dotenv(ROOT / "backend" / ".env")
    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (environment, backend/.env or .env)")
    missing = [f for f in FILES if not (ARTIFACTS_DIR / f).is_file()]
    if missing:
        raise SystemExit(f"missing files {missing} — run backend.ml.export_onnx and backend.ml.baselines first")

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    operations = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(ARTIFACTS_DIR / f)) for f in FILES]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=model_card(args.repo).encode()))
    commit = api.create_commit(repo_id=args.repo, repo_type="model", operations=operations,
                               commit_message="Upload ONNX int8 multi-task model + TF-IDF fallback")
    print(f"uploaded {len(operations)} files to https://huggingface.co/{args.repo}")
    print(f"HF_MODEL_REPO={args.repo}")
    print(f"HF_MODEL_REVISION={commit.oid}")


if __name__ == "__main__":
    main()
