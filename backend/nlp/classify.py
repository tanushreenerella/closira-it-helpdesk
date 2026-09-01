"""Backward-compatible label-only access to the DistilBERT classifier."""

try:
    from .classifier import classify_employee_message
except ImportError:  # Supports running backend modules directly as scripts.
    from classifier import classify_employee_message


def model_available() -> bool:
    try:
        classify_employee_message("")
        return True
    except Exception:
        return False


def classify(text: str) -> str:
    """Return the predicted label for callers that only need routing intent."""
    return classify_employee_message(text)["label"]
