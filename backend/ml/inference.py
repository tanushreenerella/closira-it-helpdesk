"""Runtime intent + emotion classification for the pipeline.

Models, best first (whichever is present is used):
    onnx          artifacts/onnx/       int8 ONNX export of the multi-task DistilBERT
                                        (onnxruntime + tokenizers; no torch) — the served model
    multitask     artifacts/multitask/  the PyTorch model (training / local evaluation)
    tfidf_logreg  artifacts/baseline/   TF-IDF + LogReg fallback

Serving (backend/main.py calls start_background_loading() at startup): if
HF_MODEL_REPO is set, the TF-IDF files and then the ONNX model are downloaded
from the Hugging Face Hub (pinned to HF_MODEL_REVISION; HF_TOKEN for a
private repo) into artifacts/ in a background thread. TF-IDF serves as soon
as it is loaded; ONNX replaces it when ready. Until anything is loaded,
classify_message() raises and the pipeline runs degraded (keyword rules only).

Without background loading (CLI, evaluation scripts) the best available model
is loaded on first use. CLOSIRA_CLASSIFIER=onnx|multitask|tfidf_logreg pins one.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable, TypedDict

import numpy as np

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ONNX_DIR = ARTIFACTS_DIR / "onnx"
MULTITASK_DIR = ARTIFACTS_DIR / "multitask"
BASELINE_DIR = ARTIFACTS_DIR / "baseline"
ONNX_MODEL_FILE = "model.int8.onnx"
TASKS = ("intent", "emotion")
MODEL_ORDER = ("onnx", "multitask", "tfidf_logreg")


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
    present = {
        "onnx": (ONNX_DIR / ONNX_MODEL_FILE).is_file() and (ONNX_DIR / "labels.json").is_file(),
        "multitask": (MULTITASK_DIR / "labels.json").is_file(),
        "tfidf_logreg": (BASELINE_DIR / "intent.joblib").is_file(),
    }
    return [name for name in MODEL_ORDER if present[name]]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def load_onnx(model_dir: Path | None = None, threads: int | None = None):
    """(session, tokenizer, labels) for the ONNX export — onnxruntime + tokenizers only."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    model_dir = Path(model_dir or ONNX_DIR)  # resolved at call time, not import time
    labels = json.loads((model_dir / "labels.json").read_text())
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=labels.get("max_length", 128))
    pad_id = tokenizer.token_to_id("[PAD]")
    tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
    options = ort.SessionOptions()
    threads = threads or int(os.environ.get("CLOSIRA_ORT_THREADS", "1"))
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_dir / ONNX_MODEL_FILE), options, providers=["CPUExecutionProvider"])
    return session, tokenizer, labels


def onnx_probs(session, tokenizer, texts: list[str]) -> dict[str, np.ndarray]:
    enc = tokenizer.encode_batch(texts)
    feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
    intent_logits, emotion_logits = session.run(["intent_logits", "emotion_logits"], feed)
    return {"intent": softmax(intent_logits), "emotion": softmax(emotion_logits)}


def load_predictor(name: str) -> Callable[[list[str]], list[dict]]:
    """Batch predictor: texts -> [{"intent": {label: p}, "emotion": {label: p}}]."""
    if name == "onnx":
        session, tokenizer, labels = load_onnx()

        def run(texts):
            # One message per call, exactly as served: int8 dynamic quantization
            # scales activations per input tensor, so batching would change outputs.
            out = []
            for text in texts:
                probs = onnx_probs(session, tokenizer, [text])
                out.append({t: dict(zip(labels[t], probs[t][0].tolist())) for t in TASKS})
            return out
        return run

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


# ── Active predictor ─────────────────────────────────────────────────────────
_lock = threading.Lock()
_active: dict = {"name": None, "run": None}
_status: dict = {"state": "idle", "error": None}   # idle | loading | ready | failed
_background = threading.Event()                    # set once background loading has started


def _activate(name: str) -> None:
    run = load_predictor(name)
    with _lock:
        _active["name"], _active["run"] = name, run
    print(f"[classifier] serving with {name}")


def download_from_hub(patterns: list[str]) -> None:
    """Fetch files from HF_MODEL_REPO@HF_MODEL_REVISION into artifacts/ (skips files already there)."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=os.environ["HF_MODEL_REPO"],
        revision=os.environ.get("HF_MODEL_REVISION") or None,
        token=os.environ.get("HF_TOKEN") or None,
        local_dir=str(ARTIFACTS_DIR),
        allow_patterns=patterns,
    )


def _load_in_background() -> None:
    _status["state"] = "loading"
    try:
        from_hub = bool(os.environ.get("HF_MODEL_REPO"))
        # 1) small TF-IDF fallback first, so classification starts quickly
        if from_hub:
            download_from_hub(["baseline/*"])
        if "tfidf_logreg" in available_models():
            _activate("tfidf_logreg")
        # 2) then the ONNX model, which replaces it
        if from_hub:
            download_from_hub(["onnx/*"])
        best = next((m for m in available_models() if m != "tfidf_logreg"), None)
        if best:
            _activate(best)
        _status["state"] = "ready" if _active["run"] else "failed"
        if not _active["run"]:
            _status["error"] = f"no intent/emotion model found in {ARTIFACTS_DIR}"
    except Exception as exc:  # keep serving with whatever loaded
        print(f"[classifier] background loading failed: {exc!r}")
        _status["state"], _status["error"] = ("ready" if _active["run"] else "failed"), repr(exc)


def start_background_loading() -> threading.Thread | None:
    """Load models in a background thread (called once at app startup)."""
    if os.environ.get("CLOSIRA_CLASSIFIER", "").strip() or _background.is_set():
        return None  # a pinned model loads synchronously on first use
    _background.set()
    thread = threading.Thread(target=_load_in_background, name="classifier-loader", daemon=True)
    thread.start()
    return thread


def classifier_status() -> dict:
    return {"model": _active["name"], **_status}


def _current_predictor() -> tuple[str, Callable]:
    with _lock:
        if _active["run"] is not None:
            return _active["name"], _active["run"]
    if _background.is_set():
        raise RuntimeError(f"classifier not ready ({_status['state']})")
    # No background loading (CLI / evaluation): load synchronously on first use.
    pinned = os.environ.get("CLOSIRA_CLASSIFIER", "").strip()
    models = [pinned] if pinned else available_models()
    if not models:
        message = f"No intent/emotion model found in {ARTIFACTS_DIR}"
        print(f"[classifier] {message}")
        raise RuntimeError(message)
    _activate(models[0])
    return _active["name"], _active["run"]


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
    name, run = _current_predictor()
    return to_prediction(run([text])[0], name)
