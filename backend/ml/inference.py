"""Runtime intent + emotion classification for the pipeline.

Loads the multi-task DistilBERT from artifacts/multitask/ if it exists,
otherwise the TF-IDF/LogReg baselines from artifacts/baseline/. If neither
is present, classify_message() raises and the pipeline falls back to a
neutral, degraded prediction.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypedDict

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MULTITASK_DIR = ARTIFACTS_DIR / "multitask"
BASELINE_DIR = ARTIFACTS_DIR / "baseline"
TASKS = ("intent", "emotion")


class Prediction(TypedDict):
    intent: str
    intent_confidence: float
    intent_probs: dict[str, float]
    emotion: str
    emotion_confidence: float
    emotion_probs: dict[str, float]
    degraded: bool
    model: str


def available_models() -> list[str]:
    found = []
    if (MULTITASK_DIR / "labels.json").is_file():
        found.append("multitask")
    if (BASELINE_DIR / "intent.joblib").is_file():
        found.append("tfidf_logreg")
    return found


def load_predictor(name: str) -> Callable[[list[str]], list[dict]]:
    """Batch predictor: texts -> [{"intent": {label: p}, "emotion": {label: p}}]."""
    if name == "multitask":
        from .multitask import load_model, predict
        model, tokenizer, labels = load_model(MULTITASK_DIR)

        def run(texts):
            probs = {t: predict(model, tokenizer, texts, t, "cpu") for t in TASKS}
            return [{t: dict(zip(labels[t], probs[t][i].tolist())) for t in TASKS} for i in range(len(texts))]
        return run

    if name == "tfidf_logreg":
        import joblib
        pipes = {t: joblib.load(BASELINE_DIR / f"{t}.joblib") for t in TASKS}

        def run(texts):
            probs = {t: pipe.predict_proba(texts) for t, pipe in pipes.items()}
            return [{t: dict(zip(pipes[t].classes_, probs[t][i].tolist())) for t in TASKS} for i in range(len(texts))]
        return run

    raise ValueError(f"unknown model {name!r}")


@lru_cache(maxsize=1)
def _default_predictor():
    # CLOSIRA_CLASSIFIER=multitask|tfidf_logreg pins a model; otherwise the
    # first available one (multitask before the baseline) is used.
    pinned = os.environ.get("CLOSIRA_CLASSIFIER", "").strip()
    models = [pinned] if pinned else available_models()
    if not models:
        message = f"No intent/emotion model found in {ARTIFACTS_DIR}"
        print(f"[classifier] {message}")
        raise RuntimeError(message)
    return models[0], load_predictor(models[0])


def to_prediction(probs: dict, model: str) -> Prediction:
    intent = max(probs["intent"], key=probs["intent"].get)
    emotion = max(probs["emotion"], key=probs["emotion"].get)
    return {
        "intent": intent, "intent_confidence": float(probs["intent"][intent]), "intent_probs": probs["intent"],
        "emotion": emotion, "emotion_confidence": float(probs["emotion"][emotion]), "emotion_probs": probs["emotion"],
        "degraded": False, "model": model,
    }


def classify_message(text: str) -> Prediction:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    name, run = _default_predictor()
    return to_prediction(run([text])[0], name)
