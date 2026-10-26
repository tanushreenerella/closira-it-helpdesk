"""Accuracy + macro-F1 for the multi-task DistilBERT, with majority-class baselines.

    python -m backend.ml.accuracy

Evaluation sets:
    intent_test_split               processed/intent_test.csv.gz
    emotion_test_split              processed/emotion_test.csv.gz (training distribution)
    emotion_goemotions_natural_test GoEmotions OFFICIAL test.tsv, single-label, mapped with
                                    emotion_map.json, neutral not downsampled; texts that also
                                    appear in our emotion train/val split are removed (the
                                    training build pooled GoEmotions train/dev/test before
                                    re-splitting, so most official test texts were trained on)

The processed splits must be the ones the model was trained on — rebuild with
`python -m backend.ml.data.build_datasets` (results/data_stats.json should come
out identical to the committed one). Writes results/accuracy.json.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .data.build_datasets import invert, load_json, normalize
from .data.sources import fetch_bytes, open_text
from .multitask import load_model, predict

ML_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ML_DIR / "data" / "processed"
MODEL_DIR = ML_DIR / "artifacts" / "multitask"
RESULTS_DIR = ML_DIR / "results"

TITLES = {
    "intent_test_split": "Intent — test split",
    "emotion_test_split": "Emotion — test split (training distribution)",
    "emotion_goemotions_natural_test": "Emotion — GoEmotions test, natural distribution",
}


def score(model, tokenizer, labels: dict, task: str, df: pd.DataFrame) -> dict:
    probs = predict(model, tokenizer, df["text"].astype(str).tolist(), task, "cpu")
    pred = [labels[task][i] for i in probs.argmax(axis=1)]
    gold = df["label"].tolist()
    majority = Counter(gold).most_common(1)[0]
    return {
        "rows": len(df),
        "accuracy": round(float(accuracy_score(gold, pred)), 4),
        "macro_f1": round(float(f1_score(gold, pred, labels=labels[task], average="macro", zero_division=0)), 4),
        "majority_class_baseline_accuracy": round(majority[1] / len(gold), 4),
        "majority_class": majority[0],
        "label_distribution": dict(Counter(gold).most_common()),
    }


def natural_goemotions_test() -> tuple[pd.DataFrame, dict]:
    """Official GoEmotions test, single-label, mapped; minus texts seen in our train/val."""
    names = fetch_bytes("goemotions_emotions.txt").decode().split()
    mapping = invert(load_json("emotion_map.json")["goemotions"], set(names), "goemotions")
    tsv = pd.read_csv(open_text("goemotions_test.tsv"), sep="\t", header=None, names=["text", "ids", "id"],
                      quoting=csv.QUOTE_NONE, dtype=str)
    single = tsv[tsv["ids"].str.split(",").str.len() == 1].copy()
    single["label"] = single["ids"].map(lambda i: mapping[names[int(i)]])
    mapped = single.dropna(subset=["label"])
    seen = set(pd.concat([pd.read_csv(PROCESSED_DIR / f"emotion_{s}.csv.gz") for s in ("train", "val")])["text"]
               .map(normalize))
    in_training = mapped["text"].map(normalize).isin(seen)
    counts = {"official_test_rows": len(tsv), "single_label_rows": len(single),
              "dropped_unmapped_labels": int(len(single) - len(mapped)),
              "removed_seen_in_training": int(in_training.sum())}
    return mapped[~in_training], counts


def main() -> None:
    model, tokenizer, labels = load_model(MODEL_DIR)
    natural, natural_counts = natural_goemotions_test()
    result = {
        "model": "multitask DistilBERT (backend/ml/artifacts/multitask, Colab-trained)",
        "intent_test_split": {
            "description": "processed/intent_test.csv.gz — the held-out split of the training data build "
                           "(Banking77 + CLINC150 + SNIPS + TWCS weak labels)",
            **score(model, tokenizer, labels, "intent", pd.read_csv(PROCESSED_DIR / "intent_test.csv.gz")),
        },
        "emotion_test_split": {
            "description": "processed/emotion_test.csv.gz — GoEmotions single-label (official train+dev+test "
                           "pooled, deduplicated, re-split), mapped to 6 emotions. The neutral downsampling rule "
                           "(<= 2x the largest other class) did not trigger in this build: 16,021 neutral vs a cap "
                           "of 31,388 (2 x 15,694 happy), so this split already has the natural single-label distribution.",
            **score(model, tokenizer, labels, "emotion", pd.read_csv(PROCESSED_DIR / "emotion_test.csv.gz")),
        },
        "emotion_goemotions_natural_test": {
            "description": "GoEmotions OFFICIAL test.tsv, single-label examples only, mapped to 6 emotions with "
                           "emotion_map.json, neutral NOT downsampled (natural distribution). Texts also present in "
                           "our emotion train/val split were removed (the training build pooled GoEmotions "
                           "train/dev/test before re-splitting).",
            **natural_counts,
            **score(model, tokenizer, labels, "emotion", natural),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "accuracy.json").write_text(json.dumps(result, indent=2))

    print(f"{'Evaluation set':52s} {'rows':>6s} {'accuracy':>9s} {'macro-F1':>9s} {'majority acc':>13s}")
    for key, title in TITLES.items():
        r = result[key]
        print(f"{title:52s} {r['rows']:>6d} {r['accuracy']:>9.3f} {r['macro_f1']:>9.3f} "
              f"{r['majority_class_baseline_accuracy']:>13.3f}")
    n = result["emotion_goemotions_natural_test"]
    print(f"\nnatural test: {n['single_label_rows']} single-label of {n['official_test_rows']}, "
          f"{n['dropped_unmapped_labels']} unmapped labels dropped, {n['removed_seen_in_training']} seen in "
          f"train/val removed; distribution {n['label_distribution']}")


if __name__ == "__main__":
    main()
