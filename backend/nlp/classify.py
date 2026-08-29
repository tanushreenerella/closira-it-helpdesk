"""Runtime classifier for escalation-intent labels.

The application deliberately returns ``normal`` until a locally trained model
is present; agent.py then retains its existing keyword fallback guards.
"""

from functools import lru_cache
from pathlib import Path

LABELS = ["frustrated", "explicit_human_request", "repeated_confusion", "normal"]
MODEL_DIR = Path(__file__).with_name("bert_model")


@lru_cache(maxsize=1)
def _pipeline():
    if not (MODEL_DIR / "config.json").exists():
        return None
    try:
        from transformers import pipeline
        return pipeline("text-classification", model=str(MODEL_DIR), tokenizer=str(MODEL_DIR))
    except Exception:
        return None


def model_available() -> bool:
    return _pipeline() is not None


def classify(text: str) -> str:
    """Return one of LABELS; use ``normal`` when no trained artifact exists."""
    classifier = _pipeline()
    if classifier is None:
        return "normal"
    result = classifier(text, truncation=True)[0]
    label = str(result["label"]).lower()
    if label.startswith("label_"):
        label = LABELS[int(label.split("_", 1)[1])]
    return label if label in LABELS else "normal"
