"""Serving-side classifier loading: Hub download, background loading, TF-IDF fallback, ONNX."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.ml import inference


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Each test starts with no active model and no background loading."""
    monkeypatch.setitem(inference._active, "name", None)
    monkeypatch.setitem(inference._active, "run", None)
    monkeypatch.setitem(inference._status, "state", "idle")
    monkeypatch.setitem(inference._status, "error", None)
    monkeypatch.setattr(inference, "_background", inference.threading.Event())
    monkeypatch.delenv("CLOSIRA_CLASSIFIER", raising=False)
    monkeypatch.delenv("HF_MODEL_REPO", raising=False)
    yield


def fake_predictor(name):
    def run(texts):
        return [{"intent": {"issue_report": 0.9, "chit_chat": 0.1}, "emotion": {"neutral": 0.8, "angry": 0.2}}
                for _ in texts]
    run.model = name
    return run


def test_background_loads_tfidf_first_then_onnx_from_hub(monkeypatch):
    monkeypatch.setenv("HF_MODEL_REPO", "someone/model")
    events, downloaded = [], set()

    def fake_download(patterns):
        events.append(("download", patterns[0]))
        downloaded.add(patterns[0].split("/")[0])

    def fake_available():
        return [m for m, d in (("onnx", "onnx"), ("tfidf_logreg", "baseline")) if d in downloaded]

    def fake_load(name):
        events.append(("load", name))
        return fake_predictor(name)

    monkeypatch.setattr(inference, "download_from_hub", fake_download)
    monkeypatch.setattr(inference, "available_models", fake_available)
    monkeypatch.setattr(inference, "load_predictor", fake_load)

    inference.start_background_loading().join(timeout=10)

    assert events == [("download", "baseline/*"), ("load", "tfidf_logreg"), ("download", "onnx/*"), ("load", "onnx")]
    assert inference.classifier_status() == {"model": "onnx", "state": "ready", "error": None}
    assert inference.classify_message("vpn down")["model"] == "onnx"


def test_tfidf_serves_while_onnx_is_still_loading(monkeypatch):
    monkeypatch.setenv("HF_MODEL_REPO", "someone/model")
    release = inference.threading.Event()
    seen = []

    def fake_download(patterns):
        if patterns == ["onnx/*"]:
            seen.append(inference.classify_message("my vpn drops")["model"])  # mid-load request
            release.wait(5)

    monkeypatch.setattr(inference, "download_from_hub", fake_download)
    monkeypatch.setattr(inference, "available_models", lambda: ["onnx", "tfidf_logreg"])
    monkeypatch.setattr(inference, "load_predictor", fake_predictor)
    thread = inference.start_background_loading()
    release.set()
    thread.join(timeout=10)
    assert seen == ["tfidf_logreg"]
    assert inference.classifier_status()["model"] == "onnx"


def test_not_ready_raises_so_pipeline_degrades(monkeypatch):
    block = inference.threading.Event()
    monkeypatch.setenv("HF_MODEL_REPO", "someone/model")
    monkeypatch.setattr(inference, "download_from_hub", lambda patterns: block.wait(5))
    thread = inference.start_background_loading()
    with pytest.raises(RuntimeError, match="not ready"):
        inference.classify_message("hello there")
    block.set()
    thread.join(timeout=10)


def test_failed_onnx_download_keeps_tfidf(monkeypatch):
    monkeypatch.setenv("HF_MODEL_REPO", "someone/model")

    def fake_download(patterns):
        if patterns == ["onnx/*"]:
            raise ConnectionError("hub unreachable")

    monkeypatch.setattr(inference, "download_from_hub", fake_download)
    monkeypatch.setattr(inference, "available_models", lambda: ["tfidf_logreg"])
    monkeypatch.setattr(inference, "load_predictor", fake_predictor)
    inference.start_background_loading().join(timeout=10)
    status = inference.classifier_status()
    assert status["model"] == "tfidf_logreg" and status["state"] == "ready" and "hub unreachable" in status["error"]


def test_pinned_classifier_skips_background_loading(monkeypatch):
    monkeypatch.setenv("CLOSIRA_CLASSIFIER", "tfidf_logreg")
    monkeypatch.setattr(inference, "load_predictor", fake_predictor)
    assert inference.start_background_loading() is None
    assert inference.classify_message("hi")["model"] == "tfidf_logreg"  # loads synchronously on first use


def test_download_from_hub_passes_repo_revision_and_token(monkeypatch):
    import huggingface_hub
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw))
    monkeypatch.setenv("HF_MODEL_REPO", "Tanushreenerella/closira-multitask")
    monkeypatch.setenv("HF_MODEL_REVISION", "abc123")
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    inference.download_from_hub(["onnx/*"])
    assert calls == [{"repo_id": "Tanushreenerella/closira-multitask", "revision": "abc123", "token": "hf_x",
                      "local_dir": str(inference.ARTIFACTS_DIR), "allow_patterns": ["onnx/*"]}]


# ── Real ONNX export (skipped when artifacts/onnx is absent) ─────────────────

needs_onnx = pytest.mark.skipif("onnx" not in inference.available_models(), reason="no ONNX export in artifacts/")


@needs_onnx
def test_onnx_predictor_returns_valid_distributions():
    run = inference.load_predictor("onnx")
    out = run(["my VPN keeps dropping and I am so annoyed", "thanks, that worked!"])
    for probs in out:
        for task in ("intent", "emotion"):
            assert abs(sum(probs[task].values()) - 1.0) < 1e-4
    assert max(out[1]["intent"], key=out[1]["intent"].get) == "appreciation"


@needs_onnx
def test_serving_path_does_not_import_torch():
    probe = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from backend.ml import inference\n"
        "inference.load_predictor('onnx')(['vpn keeps dropping'])\n"
        "inference.load_predictor('tfidf_logreg')(['vpn keeps dropping'])\n"
        "print('torch' in sys.modules, 'transformers' in sys.modules)" % ROOT
    )
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", probe], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip().splitlines()[-1] == "False False"


@pytest.mark.skipif(not (inference.ONNX_DIR / "model.fp32.onnx").is_file(), reason="no fp32 ONNX export")
def test_exported_graph_respects_the_padding_mask():
    # Regression: an export traced with SDPA attention ignored attention_mask, so
    # padded batch rows got different outputs than the same text on its own.
    import numpy as np
    import onnxruntime as ort
    _, tok, _ = inference.load_onnx()
    session = ort.InferenceSession(str(inference.ONNX_DIR / "model.fp32.onnx"), providers=["CPUExecutionProvider"])
    texts = ["hi", "my vpn keeps dropping every ten minutes and I cannot join any client calls today", "thanks"]
    batched = inference.onnx_probs(session, tok, texts)
    for task in ("intent", "emotion"):
        single = np.concatenate([inference.onnx_probs(session, tok, [t])[task] for t in texts])
        assert np.abs(batched[task] - single).max() < 1e-4
