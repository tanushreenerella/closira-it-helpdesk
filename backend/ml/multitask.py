"""Multi-task DistilBERT: one shared encoder, an intent head and an emotion head.

Training alternates batches between the two single-task datasets (an intent
batch, then an emotion batch, ...) so the shared encoder sees both tasks
every few steps. Each head uses class-weighted cross-entropy; optimisation is
AdamW (lr 2e-5) with linear warm-up/decay and early stopping on the mean of
the two validation macro-F1 scores.

Colab:
    !git clone <repo> && cd closira
    !pip install -q "transformers>=4.40" "torch>=2.2" scikit-learn pandas
    # upload backend/ml/data/processed/{intent,emotion}_{train,val,test}.csv.gz,
    # or run  python -m backend.ml.data.build_datasets  there first
    !python -m backend.ml.multitask --epochs 6 --batch-size 32 --fp16

Local smoke test (tiny random model, a few steps):
    python -m backend.ml.multitask --model hf-internal-testing/tiny-random-DistilBertModel \
        --limit 256 --epochs 1 --out backend/ml/artifacts/multitask_smoke --results-name multitask_smoke
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import classification_report, f1_score

ML_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ML_DIR / "data" / "processed"
RESULTS_DIR = ML_DIR / "results"
DEFAULT_OUT = ML_DIR / "artifacts" / "multitask"
DEFAULT_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 128
TASKS = ("intent", "emotion")


class MultiTaskModel(nn.Module):
    def __init__(self, encoder, n_intents: int, n_emotions: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size if hasattr(encoder.config, "hidden_size") else encoder.config.dim
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({
            "intent": nn.Linear(hidden, n_intents),
            "emotion": nn.Linear(hidden, n_emotions),
        })

    def forward(self, input_ids, attention_mask, task: str | None = None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.dropout(hidden[:, 0])  # [CLS] token
        if task is not None:
            return self.heads[task](pooled)
        return {name: head(pooled) for name, head in self.heads.items()}


def save_model(model: MultiTaskModel, tokenizer, labels: dict, out_dir: Path, base_model: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out_dir / "encoder")
    tokenizer.save_pretrained(out_dir / "encoder")
    torch.save(model.heads.state_dict(), out_dir / "heads.pt")
    (out_dir / "labels.json").write_text(json.dumps({**labels, "base_model": base_model, "max_length": MAX_LENGTH}, indent=2))


def load_model(model_dir: Path, device: str = "cpu"):
    """Load a saved MultiTaskModel. Returns (model, tokenizer, labels)."""
    from transformers import AutoModel, AutoTokenizer

    labels = json.loads((model_dir / "labels.json").read_text())
    encoder = AutoModel.from_pretrained(model_dir / "encoder")
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "encoder")
    model = MultiTaskModel(encoder, len(labels["intent"]), len(labels["emotion"]))
    model.heads.load_state_dict(torch.load(model_dir / "heads.pt", map_location=device))
    model.to(device).eval()
    return model, tokenizer, labels


# ── Data ─────────────────────────────────────────────────────────────────────
def read_split(task: str, split: str, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / f"{task}_{split}.csv.gz")
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=0)
    return df.reset_index(drop=True)


def batches(texts: list[str], ids: list[int], batch_size: int, shuffle: bool, rng: random.Random):
    order = list(range(len(texts)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        yield [texts[j] for j in idx], torch.tensor([ids[j] for j in idx])


def class_weights(ids: list[int], n: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1."""
    counts = np.bincount(ids, minlength=n).astype(float)
    weights = counts.sum() / (n * np.maximum(counts, 1))
    return torch.tensor(weights / weights.mean(), dtype=torch.float)


@torch.no_grad()
def predict(model, tokenizer, texts: list[str], task: str, device: str, batch_size: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
        out.append(torch.softmax(model(enc["input_ids"], enc["attention_mask"], task=task), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(model.heads[task].weight)))


def evaluate(model, tokenizer, data: dict, labels: dict, device: str) -> dict:
    scores = {}
    for task in TASKS:
        df = data[task]
        probs = predict(model, tokenizer, df["text"].tolist(), task, device)
        pred = probs.argmax(axis=1)
        gold = df["label"].map({l: i for i, l in enumerate(labels[task])}).to_numpy()
        scores[task] = {
            "macro_f1": float(f1_score(gold, pred, average="macro", labels=range(len(labels[task])), zero_division=0)),
            "report": classification_report(gold, pred, labels=range(len(labels[task])), target_names=labels[task],
                                             output_dict=True, zero_division=0),
        }
    return scores


# ── Training ─────────────────────────────────────────────────────────────────
def train(args) -> dict:
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = {t: read_split(t, "train", args.limit) for t in TASKS}
    val_df = {t: read_split(t, "val", args.limit and max(64, args.limit // 4)) for t in TASKS}
    test_df = {t: read_split(t, "test", args.limit and max(64, args.limit // 4)) for t in TASKS}

    labels = {
        "intent": json.loads((ML_DIR / "data" / "intent_map.json").read_text())["intents"],
        "emotion": json.loads((ML_DIR / "data" / "emotion_map.json").read_text())["emotions"],
    }
    label_ids = {t: {l: i for i, l in enumerate(labels[t])} for t in TASKS}
    train_texts = {t: train_df[t]["text"].astype(str).tolist() for t in TASKS}
    train_ids = {t: train_df[t]["label"].map(label_ids[t]).tolist() for t in TASKS}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model)
    model = MultiTaskModel(encoder, len(labels["intent"]), len(labels["emotion"])).to(device)
    loss_fns = {t: nn.CrossEntropyLoss(weight=class_weights(train_ids[t], len(labels[t])).to(device)) for t in TASKS}

    steps_per_epoch = 2 * max(math.ceil(len(train_texts[t]) / args.batch_size) for t in TASKS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps_per_epoch * args.epochs), steps_per_epoch * args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device == "cuda")

    best, best_epoch, bad_epochs, history = -1.0, -1, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, losses = time.time(), {t: [] for t in TASKS}
        # Alternate tasks batch by batch; the smaller dataset is cycled so
        # both heads get the same number of updates per epoch.
        iters = {t: batches(train_texts[t], train_ids[t], args.batch_size, True, rng) for t in TASKS}
        for step in range(steps_per_epoch):
            task = TASKS[step % 2]
            try:
                texts, ids = next(iters[task])
            except StopIteration:
                iters[task] = batches(train_texts[task], train_ids[task], args.batch_size, True, rng)
                texts, ids = next(iters[task])
            enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16 and device == "cuda"):
                logits = model(enc["input_ids"], enc["attention_mask"], task=task)
                loss = loss_fns[task](logits.float(), ids.to(device))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses[task].append(loss.item())
            if args.log_every and step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch} {task} loss {loss.item():.4f}", flush=True)

        val = evaluate(model, tokenizer, val_df, labels, device)
        score = (val["intent"]["macro_f1"] + val["emotion"]["macro_f1"]) / 2
        history.append({"epoch": epoch, "seconds": round(time.time() - started, 1),
                        **{f"train_loss_{t}": float(np.mean(losses[t])) for t in TASKS},
                        **{f"val_macro_f1_{t}": val[t]["macro_f1"] for t in TASKS}, "val_mean_macro_f1": score})
        print(json.dumps(history[-1]), flush=True)

        if score > best:
            best, best_epoch, bad_epochs = score, epoch, 0
            save_model(model, tokenizer, labels, Path(args.out), args.model)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    model, tokenizer, _ = load_model(Path(args.out), device)
    test = evaluate(model, tokenizer, test_df, labels, device)
    results = {
        "model": args.model, "device": device, "best_epoch": best_epoch, "best_val_mean_macro_f1": best,
        "config": {k: v for k, v in vars(args).items()}, "train_sizes": {t: len(train_texts[t]) for t in TASKS},
        "history": history,
        "test": {t: {"macro_f1": test[t]["macro_f1"],
                     "per_class_f1": {l: test[t]["report"][l]["f1-score"] for l in labels[t]}} for t in TASKS},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.results_name}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["test"], indent=2))
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--results-name", default="multitask")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--patience", type=int, default=2, help="epochs without val improvement before stopping")
    p.add_argument("--limit", type=int, default=None, help="subsample each train split (smoke tests)")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=200)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
