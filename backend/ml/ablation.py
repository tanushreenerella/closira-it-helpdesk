"""Ablation: 1 shared multi-task model vs 2 separate single-task models.

    python -m backend.ml.ablation [--shared DIR] [--intent DIR] [--emotion DIR] [--latency-samples 200]

Defaults compare artifacts/ablation_both (the multi-task model retrained with
the ablation runs) against artifacts/ablation_intent + artifacts/ablation_emotion,
all from `python -m backend.ml.multitask --tasks {both,intent,emotion}` with
identical splits, seed, hyperparameters and early stopping.

Reported per setup, on the test splits: intent accuracy / macro-F1, emotion
accuracy / macro-F1, frustrated F1, parameter count, and CPU latency per
message (PyTorch, 1 thread; the shared model runs ONE forward pass for both
heads, the separate setup one pass per model). Writes
results/ablation_comparison.json and prints the table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from .multitask import MAX_LENGTH, load_model, parameter_counts, predict

ML_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ML_DIR / "artifacts"
PROCESSED_DIR = ML_DIR / "data" / "processed"
RESULTS_DIR = ML_DIR / "results"


def task_metrics(model, tokenizer, labels: dict, task: str, df: pd.DataFrame, device: str) -> dict:
    probs = predict(model, tokenizer, df["text"].astype(str).tolist(), task, device)
    pred = [labels[task][i] for i in probs.argmax(axis=1)]
    gold = df["label"].tolist()
    per_class = f1_score(gold, pred, labels=labels[task], average=None, zero_division=0)
    out = {"accuracy": round(float(np.mean([g == p for g, p in zip(gold, pred)])), 4),
           "macro_f1": round(float(per_class.mean()), 4)}
    if task == "emotion":
        out["frustrated_f1"] = round(float(per_class[labels[task].index("frustrated")]), 4)
    return out


@torch.no_grad()
def latency(models: list[tuple], texts: list[str]) -> dict:
    """Per-message CPU latency; `models` = [(model, tokenizer, task or None)], all run per message."""
    def once(text):
        for model, tokenizer, task in models:
            enc = tokenizer([text], truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            model(enc["input_ids"], enc["attention_mask"], task=task)
    for t in texts[:5]:
        once(t)  # warm-up
    times = []
    for t in texts:
        start = time.perf_counter()
        once(t)
        times.append((time.perf_counter() - start) * 1000)
    return {"messages": len(texts), "mean_ms": round(float(np.mean(times)), 1),
            "p50_ms": round(float(np.percentile(times, 50)), 1), "p95_ms": round(float(np.percentile(times, 95)), 1)}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shared", default=str(ARTIFACTS_DIR / "ablation_both"))
    p.add_argument("--intent", default=str(ARTIFACTS_DIR / "ablation_intent"))
    p.add_argument("--emotion", default=str(ARTIFACTS_DIR / "ablation_emotion"))
    p.add_argument("--latency-samples", type=int, default=200)
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {t: pd.read_csv(PROCESSED_DIR / f"{t}_test.csv.gz") for t in ("intent", "emotion")}
    shared, sh_tok, sh_labels = load_model(Path(args.shared), device)
    intent_m, in_tok, in_labels = load_model(Path(args.intent), device)
    emotion_m, em_tok, em_labels = load_model(Path(args.emotion), device)
    assert set(shared.heads) == {"intent", "emotion"}, "--shared must have both heads"
    assert set(intent_m.heads) == {"intent"} and set(emotion_m.heads) == {"emotion"}, "--intent/--emotion must be single-task"

    setups = {
        "shared_1_model": {
            "intent": task_metrics(shared, sh_tok, sh_labels, "intent", data["intent"], device),
            "emotion": task_metrics(shared, sh_tok, sh_labels, "emotion", data["emotion"], device),
            "parameters": parameter_counts(shared)["total"],
        },
        "separate_2_models": {
            "intent": task_metrics(intent_m, in_tok, in_labels, "intent", data["intent"], device),
            "emotion": task_metrics(emotion_m, em_tok, em_labels, "emotion", data["emotion"], device),
            "parameters": parameter_counts(intent_m)["total"] + parameter_counts(emotion_m)["total"],
        },
    }

    # latency on CPU, one thread (as served)
    torch.set_num_threads(1)
    for m in (shared, intent_m, emotion_m):
        m.to("cpu").eval()
    texts = data["intent"]["text"].astype(str).sample(n=args.latency_samples, random_state=0).tolist()
    setups["shared_1_model"]["cpu_latency_per_message"] = latency([(shared, sh_tok, None)], texts)
    setups["separate_2_models"]["cpu_latency_per_message"] = latency(
        [(intent_m, in_tok, "intent"), (emotion_m, em_tok, "emotion")], texts)

    result = {"shared": args.shared, "intent": args.intent, "emotion": args.emotion,
              "test_rows": {t: len(d) for t, d in data.items()}, "latency_device": "cpu, 1 thread (PyTorch)",
              "setups": setups}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ablation_comparison.json").write_text(json.dumps(result, indent=2))

    s, d = setups["shared_1_model"], setups["separate_2_models"]
    rows = [
        ("Intent accuracy", s["intent"]["accuracy"], d["intent"]["accuracy"]),
        ("Intent macro-F1", s["intent"]["macro_f1"], d["intent"]["macro_f1"]),
        ("Emotion accuracy", s["emotion"]["accuracy"], d["emotion"]["accuracy"]),
        ("Emotion macro-F1", s["emotion"]["macro_f1"], d["emotion"]["macro_f1"]),
        ("Frustrated F1", s["emotion"]["frustrated_f1"], d["emotion"]["frustrated_f1"]),
        ("Parameters (M)", round(s["parameters"] / 1e6, 2), round(d["parameters"] / 1e6, 2)),
        ("CPU latency / message, mean (ms)", s["cpu_latency_per_message"]["mean_ms"], d["cpu_latency_per_message"]["mean_ms"]),
        ("CPU latency / message, p95 (ms)", s["cpu_latency_per_message"]["p95_ms"], d["cpu_latency_per_message"]["p95_ms"]),
    ]
    print(f"| {'Metric':34s} | {'1 shared model':>15s} | {'2 separate models':>18s} |")
    print(f"|{'-' * 36}|{'-' * 17}|{'-' * 20}|")
    for name, a, b in rows:
        print(f"| {name:34s} | {a:>15} | {b:>18} |")


if __name__ == "__main__":
    main()
