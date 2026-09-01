"""DistilBERT inference for Closira escalation-intent triage."""

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

MODEL_DIR = Path(__file__).with_name("model")
MAX_LENGTH = 128


class IntentPrediction(TypedDict):
    label: str
    confidence: float


@lru_cache(maxsize=1)
def _load_model():
    """Load the saved local model once per application process."""
    if not (MODEL_DIR / "config.json").is_file():
        raise RuntimeError(f"DistilBERT model is missing from {MODEL_DIR}")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model, torch


def classify_employee_message(message: str) -> IntentPrediction:
    """Return the fine-tuned intent label and its softmax confidence."""
    if not isinstance(message, str):
        raise TypeError("message must be a string")
    tokenizer, model, torch = _load_model()
    inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=1)
    index = int(torch.argmax(probabilities, dim=1).item())
    return {
        "label": str(model.config.id2label[index]).lower(),
        "confidence": float(probabilities[0, index].item()),
    }
