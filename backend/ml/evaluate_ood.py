"""Out-of-domain evaluation on the 616 legacy IT-helpdesk messages.

    python -m backend.ml.evaluate_ood [--model multitask|tfidf_logreg|all]

The legacy set (data/ood/it_messages_616.csv) is labelled with the old
escalation classes — normal / frustrated / explicit_human_request /
repeated_confusion — not with the new intents or emotions, and none of it
was used for training. So it is scored two ways:

1. Escalation (binary): gold = label != "normal". Compared for
     keyword_rules  - router hard-override flags only
     model_only     - router on intent/emotion/risk with no flags
     router         - flags + model (what the pipeline does in the faq stage)
2. Label alignment: how each legacy class is distributed over predicted
   emotions and intents (e.g. how much of "frustrated" lands on
   angry/frustrated).

Writes results/ood_<model>.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .inference import available_models, load_predictor, to_prediction
from ..router import (HARD_OVERRIDE_FLAGS, INTENT_CONFIDENCE_THRESHOLD, UNCERTAIN_INTENT, emotion_only_risk,
                      keyword_flags, risk_score, route)

ML_DIR = Path(__file__).resolve().parent
OOD_PATH = ML_DIR / "data" / "ood" / "it_messages_616.csv"
RESULTS_DIR = ML_DIR / "results"

# Which predicted emotions count as "consistent" with each legacy label.
CONSISTENT_EMOTIONS = {
    "frustrated": {"angry", "frustrated"},
    "repeated_confusion": {"confused", "frustrated"},
    "explicit_human_request": None,  # an explicit request carries no particular emotion
    "normal": {"neutral", "happy", "confused"},
}


def binary(gold: list[bool], pred: list[bool]) -> dict:
    p, r, f, _ = precision_recall_fscore_support(gold, pred, average="binary", pos_label=True, zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f),
            "accuracy": float(sum(g == q for g, q in zip(gold, pred)) / len(gold))}


def recall_by_legacy_label(df: pd.DataFrame, col: str) -> dict:
    return {label: float(part[col].mean()) for label, part in df.groupby("label")}


def evaluate(model: str) -> dict:
    df = pd.read_csv(OOD_PATH)
    preds = [to_prediction(p, model) for p in load_predictor(model)(df["text"].astype(str).tolist())]
    df["intent_raw"] = [p["intent"] for p in preds]
    # Same rule as the pipeline: a low-confidence intent is "uncertain" and
    # left out of routing and risk.
    certain = [p["intent_confidence"] >= INTENT_CONFIDENCE_THRESHOLD for p in preds]
    df["intent"] = [p["intent"] if ok else UNCERTAIN_INTENT for p, ok in zip(preds, certain)]
    df["emotion"] = [p["emotion"] for p in preds]
    flags = [keyword_flags(t) for t in df["text"]]
    df["risk"] = [risk_score(p["intent"], p["emotion"], p["intent_probs"], p["emotion_probs"]) if ok
                  else emotion_only_risk(p["emotion"], p["emotion_probs"], p["emotion_confidence"], f)
                  for p, ok, f in zip(preds, certain, flags)]
    df["kw"] = [any(v for k, v in f.items() if k in HARD_OVERRIDE_FLAGS) for f in flags]
    df["model_only"] = [route(i, e, r, {})[0] == "escalate" for i, e, r in zip(df["intent"], df["emotion"], df["risk"])]
    df["router"] = [route(i, e, r, f)[0] == "escalate" for i, e, r, f in zip(df["intent"], df["emotion"], df["risk"], flags)]
    gold = (df["label"] != "normal").tolist()

    alignment = {}
    for label, part in df.groupby("label"):
        consistent = CONSISTENT_EMOTIONS.get(label)
        alignment[label] = {
            "n": len(part),
            "emotion_distribution": dict(Counter(part["emotion"]).most_common()),
            "intent_distribution": dict(Counter(part["intent"]).most_common()),
            "raw_intent_distribution": dict(Counter(part["intent_raw"]).most_common()),
            "emotion_consistent_rate": float(part["emotion"].isin(consistent).mean()) if consistent else None,
            "mean_risk": float(part["risk"].mean()),
        }

    return {
        "model": model, "n": len(df), "legacy_labels": dict(Counter(df["label"])),
        "uncertain_intent_rate": float((df["intent"] == UNCERTAIN_INTENT).mean()),
        "intent_confidence_threshold": INTENT_CONFIDENCE_THRESHOLD,
        "escalation": {name: {**binary(gold, df[col].tolist()), "escalate_rate_by_legacy_label": recall_by_legacy_label(df, col)}
                       for name, col in (("keyword_rules", "kw"), ("model_only", "model_only"), ("router", "router"))},
        "label_alignment": alignment,
        "examples_missed_by_router": df[(df["label"] != "normal") & ~df["router"]].head(15)[["text", "label", "intent", "emotion", "risk"]].to_dict("records"),
        "examples_false_alarm": df[(df["label"] == "normal") & df["router"]].head(15)[["text", "intent", "emotion", "risk"]].to_dict("records"),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="all")
    args = p.parse_args(argv)
    models = available_models() if args.model == "all" else [args.model]
    if not models:
        raise SystemExit("no trained model in backend/ml/artifacts — run backend.ml.baselines first")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for model in models:
        result = evaluate(model)
        (RESULTS_DIR / f"ood_{model}.json").write_text(json.dumps(result, indent=2))
        print(f"\n== {model}")
        for name, m in result["escalation"].items():
            print(f"  {name:14s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                  f"by-label={ {k: round(v, 2) for k, v in m['escalate_rate_by_legacy_label'].items()} }")
        for label, a in result["label_alignment"].items():
            print(f"  {label:24s} emotion-consistent={a['emotion_consistent_rate']} mean_risk={a['mean_risk']:.2f} "
                  f"top emotions={list(a['emotion_distribution'].items())[:3]}")


if __name__ == "__main__":
    main()
