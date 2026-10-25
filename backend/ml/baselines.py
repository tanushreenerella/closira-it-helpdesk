"""Baselines for the intent and emotion tasks.

    python -m backend.ml.baselines

1. keyword rules  — hand-written lexicons, no training
2. TF-IDF + LogReg — word 1-2grams + char 3-5grams, class_weight="balanced"

Both are scored on the held-out test split (macro-F1 + per-class F1).
Writes results/baselines.json and results/baselines_per_class_f1.csv, and
saves the fitted TF-IDF pipelines to artifacts/baseline/ — the runtime
fallback when the multi-task model hasn't been trained.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

ML_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ML_DIR / "data" / "processed"
RESULTS_DIR = ML_DIR / "results"
BASELINE_DIR = ML_DIR / "artifacts" / "baseline"
TASKS = ("intent", "emotion")


def labels_for(task: str) -> list[str]:
    name = "intent_map.json" if task == "intent" else "emotion_map.json"
    return json.loads((ML_DIR / "data" / name).read_text())["intents" if task == "intent" else "emotions"]


# ── Keyword rules ────────────────────────────────────────────────────────────
def _rx(*words: str) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(words) + r")\b", re.I)


# Checked in order; first match wins.
INTENT_RULES = [
    ("complaint", _rx("worst", "terrible", "awful", "horrible", "ridiculous", "unacceptable", "useless",
                      "pathetic", "disgusted", "fed up", "sick of", "rip ?off", "never again", "disappointed")),
    ("appreciation", _rx("thanks?", "thank you", "thx", "appreciate\\w*", "great service", "awesome", "you rock",
                         "kudos")),
    ("feedback", _rx("suggest\\w*", "feedback", "you should", "would be (nice|great)", "feature request",
                     "please add", "i wish")),
    ("issue_report", _rx("not working", "doesn'?t work", "won'?t", "can'?t", "cannot", "unable", "failed",
                         "fails?", "error", "broken", "declined", "lost", "stolen", "missing", "wrong",
                         "charged twice", "still pending", "hasn'?t arrived", "not received", "blocked", "stuck")),
    ("chit_chat", re.compile(r"^\s*(hi|hello|hey|good (morning|evening|night)|bye|goodbye|yes|no|maybe|ok(ay)?|"
                             r"are you a (bot|robot)|what('?s| is) your name|tell me a joke|how are you)\b", re.I)),
    ("action_request", re.compile(r"^\s*(please\s+)?(cancel|book|play|add|set|order|change|reset|activate|"
                                  r"transfer|send|turn|schedule|remind|put|make|call|text|update|freeze|"
                                  r"rate|give|get me|find)\b|\b(i want to|i need to|i'?d like to|can you|could you)\b",
                                  re.I)),
]

EMOTION_RULES = [
    ("angry", _rx("angry", "furious", "hate", "pissed", "disgust\\w*", "outraged", "fuck\\w*", "damn", "wtf",
                  "screw (you|this)")),
    ("frustrated", _rx("annoy\\w*", "frustrat\\w*", "ugh+", "ridiculous", "seriously", "fed up", "sick of",
                       "stupid", "useless", "come on", "irritat\\w*")),
    ("confused", _rx("confus\\w*", "don'?t understand", "not sure", "what do you mean", "huh", "unclear",
                     "makes no sense", "i don'?t get", "lost")),
    ("sad", _rx("sad", "sorry", "miss(ing)?", "unfortunately", "disappoint\\w*", "depress\\w*", "cry\\w*", "heartbr\\w*",
                "upset", "rip")),
    ("happy", _rx("thanks?", "thank you", "love", "great", "awesome", "amazing", "happy", "glad", "lol", "haha\\w*",
                  "congrat\\w*", "beautiful", "best", "nice", "cool", "good")),
]


def keyword_predict(text: str, task: str) -> str:
    rules = INTENT_RULES if task == "intent" else EMOTION_RULES
    default = "information_request" if task == "intent" else "neutral"
    return next((label for label, rx in rules if rx.search(text)), default)


# ── TF-IDF + LogReg ──────────────────────────────────────────────────────────
def tfidf_logreg() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200_000)),
    ])
    return Pipeline([("features", features),
                     ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))])


def score(gold, pred, labels: list[str]) -> dict:
    report = classification_report(gold, pred, labels=labels, output_dict=True, zero_division=0)
    return {"macro_f1": float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)),
            "accuracy": float(report["accuracy"]) if "accuracy" in report else float((pd.Series(gold) == pd.Series(pred)).mean()),
            "per_class_f1": {l: float(report[l]["f1-score"]) for l in labels},
            "support": {l: int(report[l]["support"]) for l in labels}}


def main() -> None:
    results, rows = {}, []
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        labels = labels_for(task)
        train = pd.read_csv(PROCESSED_DIR / f"{task}_train.csv.gz")
        val = pd.read_csv(PROCESSED_DIR / f"{task}_val.csv.gz")
        test = pd.read_csv(PROCESSED_DIR / f"{task}_test.csv.gz")
        for df in (train, val, test):
            df["text"] = df["text"].astype(str)

        kw = score(test["label"], [keyword_predict(t, task) for t in test["text"]], labels)

        pipe = tfidf_logreg().fit(pd.concat([train["text"], val["text"]]), pd.concat([train["label"], val["label"]]))
        joblib.dump(pipe, BASELINE_DIR / f"{task}.joblib", compress=3)
        lr = score(test["label"], pipe.predict(test["text"]), labels)

        results[task] = {"test_rows": len(test), "keyword_rules": kw, "tfidf_logreg": lr}
        for label in labels:
            rows.append({"task": task, "label": label, "support": lr["support"][label],
                         "keyword_f1": round(kw["per_class_f1"][label], 4), "tfidf_logreg_f1": round(lr["per_class_f1"][label], 4)})
        rows.append({"task": task, "label": "MACRO", "support": len(test),
                     "keyword_f1": round(kw["macro_f1"], 4), "tfidf_logreg_f1": round(lr["macro_f1"], 4)})
        print(f"{task}: keyword macro-F1={kw['macro_f1']:.3f}  tfidf_logreg macro-F1={lr['macro_f1']:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "baselines.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "baselines_per_class_f1.csv", index=False)


if __name__ == "__main__":
    main()
