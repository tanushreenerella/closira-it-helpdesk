"""Build the intent and emotion training sets.

    python -m backend.ml.data.build_datasets [--skip-twcs] [--twcs-max-rows N]

Intent  : Banking77 + CLINC150 + SNIPS mapped to 7 intents (intent_map.json),
          plus TWCS customer tweets weakly labelled complaint / feedback /
          appreciation by keyword rules (twcs_rules.json).
Emotion : GoEmotions single-label examples mapped to 6 emotions
          (emotion_map.json); neutral downsampled to <= 2x the largest class.

Both sets are deduplicated on normalised text and split 80/10/10 stratified
by label. Outputs (gzipped CSV, text,label,source,source_label):
    processed/{intent,emotion}_{train,val,test}.csv.gz
    review/twcs_review_200.csv      (manual check of TWCS weak labels)
    ../results/data_stats.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .sources import fetch_bytes, open_stream, open_text

DATA_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = DATA_DIR / "processed"
REVIEW_DIR = DATA_DIR / "review"
RESULTS_DIR = DATA_DIR.parent / "results"
SEED = 42


def load_json(name: str) -> dict:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def invert(grouped: dict, source_labels: set[str], source: str) -> dict[str, str | None]:
    """{target: [labels]} -> {label: target}; _drop maps to None. Fails on gaps."""
    mapping: dict[str, str | None] = {}
    for target, labels in grouped.items():
        for label in labels:
            if label in mapping:
                raise ValueError(f"{source}: {label!r} mapped twice")
            mapping[label] = None if target == "_drop" else target
    missing = source_labels - mapping.keys()
    if missing:
        raise ValueError(f"{source}: unmapped labels {sorted(missing)}")
    return mapping


def cap_per_group(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    """Random sample of at most n rows per value of col (keeps col)."""
    return pd.concat([g.sample(n=min(n, len(g)), random_state=SEED) for _, g in df.groupby(col)])


def normalize(text: str) -> str:
    """Dedupe key: lowercase, strip punctuation and whitespace runs."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text).lower())).strip()


# ── Intent sources ───────────────────────────────────────────────────────────
def banking77() -> pd.DataFrame:
    frames = [pd.read_csv(open_text(f"banking77_{s}.csv")) for s in ("train", "test")]
    df = pd.concat(frames).rename(columns={"category": "source_label"})
    return df.assign(source="banking77")[["text", "source", "source_label"]]


def clinc150() -> pd.DataFrame:
    data = json.loads(fetch_bytes("clinc150_full.json"))
    rows = [(t, l) for split in ("train", "val", "test") for t, l in data[split]]
    rows += [(t, "oos") for split in ("oos_train", "oos_val", "oos_test") for t, _ in data[split]]
    return pd.DataFrame(rows, columns=["text", "source_label"]).assign(source="clinc150")


def snips() -> pd.DataFrame:
    rows = []
    for s in ("train", "valid", "test"):
        texts = fetch_bytes(f"snips_{s}_seq.in").decode().splitlines()
        labels = fetch_bytes(f"snips_{s}_label").decode().splitlines()
        rows += list(zip(texts, labels))
    return pd.DataFrame(rows, columns=["text", "source_label"]).assign(source="snips")


def map_intents(df: pd.DataFrame, intent_map: dict) -> pd.DataFrame:
    out = []
    for source, part in df.groupby("source"):
        mapping = invert(intent_map[source], set(part["source_label"]), source)
        part = part.assign(label=part["source_label"].map(mapping)).dropna(subset=["label"])
        # Cap each source label so large single-domain sources (SNIPS has
        # ~2k per label) don't swamp the rest.
        out.append(cap_per_group(part, "source_label", intent_map["max_per_source_label"]))
    return pd.concat(out)


# ── TWCS weak labels ─────────────────────────────────────────────────────────
HANDLE_URL = re.compile(r"@\w+|https?://\S+")


def clean_tweet(text: str) -> str:
    return re.sub(r"\s+", " ", HANDLE_URL.sub(" ", str(text))).strip()


def twcs_label(text: str, rules: dict, compiled: dict, exclude: list) -> tuple[str | None, str]:
    if any(p.search(text) for p in exclude):
        return None, ""
    hits = {label: next((p.pattern for p in pats if p.search(text)), None) for label, pats in compiled.items()}
    hits = {k: v for k, v in hits.items() if v}
    # "thanks for nothing" style sarcasm is a complaint even though it
    # contains appreciation words — complaint wins over appreciation only.
    if set(hits) == {"complaint", "appreciation"}:
        hits.pop("appreciation")
    if len(hits) != 1:
        return None, ""
    (label, pattern), = hits.items()
    return label, pattern


TWCS_CACHE = PROCESSED_DIR / "twcs_labelled.csv.gz"


def twcs(max_rows: int | None) -> pd.DataFrame:
    """Stream inbound customer tweets and keep those a keyword rule labels.

    The full scan streams ~500 MB, so the labelled tweets (before per-class
    capping) are cached in processed/. On a cached run the current rules are
    re-applied to those tweets — exact if rules were only tightened; delete
    the cache after loosening a rule."""
    rules = load_json("twcs_rules.json")
    compiled = {label: [re.compile(p, re.I) for p in pats] for label, pats in rules["rules"].items()}
    exclude = [re.compile(p, re.I) for p in rules["exclude"]]
    if TWCS_CACHE.is_file() and not max_rows:
        df = pd.read_csv(TWCS_CACHE)
        relabelled = [twcs_label(t, rules, compiled, exclude) for t in df["text"].astype(str)]
        df = df.assign(label=[l for l, _ in relabelled], source_label=[p for _, p in relabelled]).dropna(subset=["label"])
        return cap_per_group(df, "label", rules["max_per_class"])
    kept, seen = [], 0
    with open_stream("twcs.csv") as stream:
        reader = pd.read_csv(io.TextIOWrapper(stream, encoding="utf-8"), usecols=["tweet_id", "inbound", "text"],
                             chunksize=100_000, dtype={"text": str})
        for chunk in reader:
            chunk = chunk[chunk["inbound"].astype(str) == "True"]
            for tweet_id, text in zip(chunk["tweet_id"], chunk["text"]):
                text = clean_tweet(text)
                if len(text.split()) < rules["min_words"]:
                    continue
                label, pattern = twcs_label(text, rules, compiled, exclude)
                if label:
                    kept.append((text, "twcs", pattern, label, tweet_id))
            seen += len(chunk)
            print(f"[twcs] {seen} inbound tweets scanned, {len(kept)} labelled", flush=True)
            if max_rows and seen >= max_rows:
                break
    df = pd.DataFrame(kept, columns=["text", "source", "source_label", "label", "tweet_id"])
    df = df.drop_duplicates(subset="text")
    if not max_rows:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TWCS_CACHE, index=False)
    return cap_per_group(df, "label", rules["max_per_class"])


def export_twcs_review(df: pd.DataFrame, n: int = 200) -> None:
    per_class = n // df["label"].nunique()
    sample = cap_per_group(df, "label", per_class)
    if len(sample) < n:
        rest = df.drop(sample.index)
        sample = pd.concat([sample, rest.sample(n=min(n - len(sample), len(rest)), random_state=SEED)])
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    sample.assign(manual_label="", notes="")[["tweet_id", "text", "label", "source_label", "manual_label", "notes"]] \
        .rename(columns={"label": "auto_label", "source_label": "matched_rule"}) \
        .to_csv(REVIEW_DIR / "twcs_review_200.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)


# ── Emotion source ───────────────────────────────────────────────────────────
def goemotions(emotion_map: dict) -> pd.DataFrame:
    names = fetch_bytes("goemotions_emotions.txt").decode().split()
    mapping = invert(emotion_map["goemotions"], set(names), "goemotions")
    rows = []
    for s in ("train", "dev", "test"):
        tsv = pd.read_csv(open_text(f"goemotions_{s}.tsv"), sep="\t", header=None, names=["text", "ids", "id"],
                          quoting=csv.QUOTE_NONE, dtype=str)
        for text, ids in zip(tsv["text"], tsv["ids"]):
            ids = ids.split(",")
            if len(ids) == 1:  # single-label only
                rows.append((text, names[int(ids[0])]))
    df = pd.DataFrame(rows, columns=["text", "source_label"]).assign(source="goemotions")
    return df.assign(label=df["source_label"].map(mapping)).dropna(subset=["label"])


def downsample_neutral(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    counts = df["label"].value_counts()
    cap = int(ratio * counts.drop("neutral").max())
    neutral = df[df["label"] == "neutral"]
    if len(neutral) > cap:
        neutral = neutral.sample(n=cap, random_state=SEED)
    return pd.concat([df[df["label"] != "neutral"], neutral])


# ── Shared ───────────────────────────────────────────────────────────────────
def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicates (on normalised text); drop texts whose
    duplicates disagree on label entirely, since they can't be trusted."""
    df = df.assign(key=df["text"].map(normalize))
    df = df[df["key"].str.len() > 0]
    n_labels = df.groupby("key")["label"].nunique()
    conflicting = set(n_labels[n_labels > 1].index)
    before = len(df)
    df = df[~df["key"].isin(conflicting)].drop_duplicates(subset="key")
    return df.drop(columns="key"), {"rows_before": before, "conflicting_texts_dropped": len(conflicting),
                                    "rows_after": len(df)}


def split_and_write(df: pd.DataFrame, task: str) -> dict:
    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        part[["text", "label", "source", "source_label"]].to_csv(PROCESSED_DIR / f"{task}_{name}.csv.gz", index=False)
        stats[name] = {"rows": len(part), "labels": dict(Counter(part["label"]))}
    return stats


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-twcs", action="store_true", help="build without TWCS (no complaint/feedback/appreciation data)")
    p.add_argument("--twcs-max-rows", type=int, default=None, help="stop scanning TWCS after N inbound tweets")
    args = p.parse_args(argv)

    intent_map, emotion_map = load_json("intent_map.json"), load_json("emotion_map.json")
    stats: dict = {}

    intents = map_intents(pd.concat([banking77(), clinc150(), snips()]), intent_map)
    if not args.skip_twcs:
        tw = twcs(args.twcs_max_rows)
        export_twcs_review(tw)
        stats["twcs_weak_labels"] = dict(Counter(tw["label"]))
        intents = pd.concat([intents, tw.drop(columns="tweet_id")])
    intents, stats["intent_dedupe"] = dedupe(intents)
    stats["intent_by_source"] = {f"{s}/{l}": n for (s, l), n in intents.groupby(["source", "label"]).size().items()}
    stats["intent_splits"] = split_and_write(intents, "intent")

    emotions = goemotions(emotion_map)
    stats["emotion_before_downsample"] = dict(Counter(emotions["label"]))
    emotions = downsample_neutral(emotions, emotion_map["neutral_max_ratio"])
    emotions, stats["emotion_dedupe"] = dedupe(emotions)
    stats["emotion_splits"] = split_and_write(emotions, "emotion")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "intent_by_source"}, indent=2))


if __name__ == "__main__":
    main()
