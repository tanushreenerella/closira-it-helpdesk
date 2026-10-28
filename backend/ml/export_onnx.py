"""Export the multi-task DistilBERT to ONNX (int8 dynamic quantization) and verify it.

    python -m backend.ml.export_onnx [--skip-verify] [--latency-samples 200]

Export: one graph, inputs input_ids / attention_mask (dynamic batch + length),
outputs intent_logits / emotion_logits (both heads). The fp32 graph is
quantized with onnxruntime dynamic int8 quantization. artifacts/onnx/ gets
model.int8.onnx + tokenizer.json + labels.json — everything serving needs
(onnxruntime + tokenizers, no torch).

Verification vs the PyTorch model (needs requirements-train.txt):
    tokenizer parity      tokenizers vs transformers input ids on the test texts
    prediction agreement  argmax agreement + max prob difference, intent and emotion test splits
    test metrics          accuracy / macro-F1 for both
    router metrics        held-out half of the 616 messages at the current thresholds
    size, latency         file sizes; single-message CPU latency (1 thread, as served)
    memory                resident memory of a serving-style process (ONNX + TF-IDF, no torch)
Writes results/onnx_export.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parent
MULTITASK_DIR = ML_DIR / "artifacts" / "multitask"
ONNX_DIR = ML_DIR / "artifacts" / "onnx"
PROCESSED_DIR = ML_DIR / "data" / "processed"
RESULTS_DIR = ML_DIR / "results"
FP32_FILE = "model.fp32.onnx"
INT8_FILE = "model.int8.onnx"


def export(opset: int = 17) -> None:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from .multitask import load_model

    model, _tokenizer, labels = load_model(MULTITASK_DIR)
    # SDPA attention drops the padding mask when traced (the exported graph then
    # attends to [PAD] tokens); eager attention exports the mask correctly.
    model.encoder.config._attn_implementation = "eager"

    class BothHeads(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            pooled = self.m.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            return self.m.heads["intent"](pooled), self.m.heads["emotion"](pooled)

    wrapper = BothHeads(model).eval()
    # trace with a padded row so the masking path is part of the graph
    ids = torch.ones(2, 16, dtype=torch.long)
    mask = torch.ones(2, 16, dtype=torch.long)
    mask[1, 10:] = 0
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    fp32, int8 = ONNX_DIR / FP32_FILE, ONNX_DIR / INT8_FILE
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (ids, mask), str(fp32),
            input_names=["input_ids", "attention_mask"], output_names=["intent_logits", "emotion_logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                          "intent_logits": {0: "batch"}, "emotion_logits": {0: "batch"}},
            opset_version=opset, dynamo=False,
        )
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    shutil.copy(MULTITASK_DIR / "encoder" / "tokenizer.json", ONNX_DIR / "tokenizer.json")
    (ONNX_DIR / "labels.json").write_text(json.dumps({**labels, "source": "multitask", "quantization": "dynamic int8"},
                                                     indent=2))
    print(f"exported {fp32.name} ({fp32.stat().st_size / 1e6:.1f} MB) -> {int8.name} ({int8.stat().st_size / 1e6:.1f} MB)")


def _mb(path: Path) -> float:
    if path.is_dir():
        return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1)
    return round(path.stat().st_size / 1e6, 1)


def _latency(fn, texts: list[str]) -> dict:
    fn(texts[:5])  # warm-up
    times = []
    for t in texts:
        start = time.perf_counter()
        fn([t])
        times.append((time.perf_counter() - start) * 1000)
    return {"messages": len(texts), "mean_ms": round(float(np.mean(times)), 1),
            "p50_ms": round(float(np.percentile(times, 50)), 1), "p95_ms": round(float(np.percentile(times, 95)), 1)}


SERVING_MEMORY_PROBE = """
import json, os, sys, psutil
sys.path.insert(0, sys.argv[1])
from backend.ml import inference
p = psutil.Process()
base = p.memory_info().rss
run_tfidf = inference.load_predictor("tfidf_logreg"); run_tfidf(["warm up"])
with_tfidf = p.memory_info().rss
run_onnx = inference.load_predictor("onnx"); run_onnx(["my vpn keeps dropping and I am annoyed"])
both = p.memory_info().rss
print(json.dumps({"python_baseline_mb": round(base / 1e6), "after_tfidf_mb": round(with_tfidf / 1e6),
                  "after_tfidf_and_onnx_mb": round(both / 1e6), "torch_imported": "torch" in sys.modules,
                  "transformers_imported": "transformers" in sys.modules}))
"""


def verify(latency_samples: int) -> dict:
    import pandas as pd
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer

    from . import calibrate
    from .inference import load_onnx, onnx_probs
    from .multitask import load_model, predict

    torch.set_num_threads(1)
    model, hf_tok, labels = load_model(MULTITASK_DIR)
    session, tok, _ = load_onnx(ONNX_DIR, threads=1)
    data = {t: pd.read_csv(PROCESSED_DIR / f"{t}_test.csv.gz") for t in ("intent", "emotion")}

    # tokenizer parity: the served tokenizer must produce the training tokenizer's ids
    sample = pd.concat([data["intent"]["text"], data["emotion"]["text"]]).astype(str).tolist()
    hf_ids = AutoTokenizer.from_pretrained(MULTITASK_DIR / "encoder")(sample, truncation=True,
                                                                     max_length=labels["max_length"])["input_ids"]
    tok.no_padding()
    served_ids = [e.ids for e in tok.encode_batch(sample)]
    tok.enable_padding(pad_id=tok.token_to_id("[PAD]"), pad_token="[PAD]")
    parity = float(np.mean([a == b for a, b in zip(hf_ids, served_ids)]))

    # padding invariance of the exported graph (fp32, so quantization can't mask it):
    # a padded batch must give the same outputs as one message at a time
    import onnxruntime as ort
    fp32_session = ort.InferenceSession(str(ONNX_DIR / FP32_FILE), providers=["CPUExecutionProvider"])
    probe = sample[:64]
    batched = onnx_probs(fp32_session, tok, probe)["emotion"]
    single = np.concatenate([onnx_probs(fp32_session, tok, [t])["emotion"] for t in probe])
    padding_diff = float(np.abs(batched - single).max())
    del fp32_session

    result = {"tokenizer_parity": round(parity, 4),
              "padding_invariance_fp32_max_prob_diff": round(padding_diff, 6), "tasks": {}}
    for task, df in data.items():
        texts, gold = df["text"].astype(str).tolist(), df["label"].tolist()
        p_torch = predict(model, hf_tok, texts, task, "cpu")
        # one message per call, as served (int8 activation scales depend on the input batch)
        p_onnx = np.concatenate([onnx_probs(session, tok, [t])[task] for t in texts])
        pred_torch = [labels[task][i] for i in p_torch.argmax(1)]
        pred_onnx = [labels[task][i] for i in p_onnx.argmax(1)]
        result["tasks"][task] = {
            "rows": len(df),
            "prediction_agreement": round(float(np.mean(p_torch.argmax(1) == p_onnx.argmax(1))), 4),
            "max_abs_prob_diff": round(float(np.abs(p_torch - p_onnx).max()), 4),
            "pytorch": {"accuracy": round(float(accuracy_score(gold, pred_torch)), 4),
                        "macro_f1": round(float(f1_score(gold, pred_torch, labels=labels[task], average="macro")), 4)},
            "onnx_int8": {"accuracy": round(float(accuracy_score(gold, pred_onnx)), 4),
                          "macro_f1": round(float(f1_score(gold, pred_onnx, labels=labels[task], average="macro")), 4)},
        }

    # router: held-out half of the 616 messages at the current thresholds
    df = pd.read_csv(calibrate.OOD_PATH)
    _, held = train_test_split(df, test_size=0.5, stratify=df["label"], random_state=calibrate.SEED)
    gold = (held["label"] != "normal").tolist()
    result["router_held_out"] = {
        name: calibrate.metrics(calibrate.featurize(held["text"].astype(str).tolist(), name), gold,
                                held["label"].tolist(), **calibrate.DEFAULTS)
        for name in ("multitask", "onnx")
    }

    result["size_mb"] = {"pytorch_model_dir": _mb(MULTITASK_DIR), "onnx_fp32": _mb(ONNX_DIR / FP32_FILE),
                         "onnx_int8": _mb(ONNX_DIR / INT8_FILE), "served_onnx_dir_without_fp32":
                             round(_mb(ONNX_DIR) - _mb(ONNX_DIR / FP32_FILE), 1)}

    lat_texts = data["intent"]["text"].astype(str).sample(n=latency_samples, random_state=0).tolist()
    result["latency_single_message_1_thread"] = {
        "pytorch": _latency(lambda t: [predict(model, hf_tok, t, k, "cpu") for k in ("intent", "emotion")], lat_texts),
        "onnx_int8": _latency(lambda t: onnx_probs(session, tok, t), lat_texts),
    }

    probe = subprocess.run([sys.executable, "-W", "ignore", "-c", SERVING_MEMORY_PROBE, str(ML_DIR.parents[1])],
                           capture_output=True, text=True, timeout=600)
    result["serving_process_memory"] = json.loads(probe.stdout.strip().splitlines()[-1]) if probe.returncode == 0 \
        else {"error": probe.stderr[-500:]}
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-export", action="store_true", help="verify an existing export")
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--latency-samples", type=int, default=200)
    args = p.parse_args(argv)
    if not args.skip_export:
        export()
    if not args.skip_verify:
        result = verify(args.latency_samples)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "onnx_export.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
