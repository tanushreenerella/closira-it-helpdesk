"""Rule-based per-turn escalation labels for ABCD (asappresearch/abcd).

    python -m backend.ml.data.build_abcd

ABCD has no escalation annotation, so each CUSTOMER turn is weakly labelled
escalate=1 when any rule fires:
    explicit_human   - asks for a manager / supervisor / human / real person
    anger            - anger or frustration language
    repeated_failure - "still not working", "already tried", "again" + negation
    agent_escalation - the agent's reply (before the next customer turn)
                       mentions a manager, supervisor, escalation or transfer
These are heuristics — review/abcd_review_300.csv exports 300 whole
conversations (150 with >=1 positive turn, 150 without) for manual checking.

Split is by conversation ID (md5 bucket, 80/10/10), so no conversation spans
splits. Outputs:
    processed/abcd_turns_{train,val,test}.csv.gz
    review/abcd_review_300.csv
    ../results/abcd_stats.json
No conversation-level model is trained on this yet.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .sources import fetch_bytes

DATA_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = DATA_DIR / "processed"
REVIEW_DIR = DATA_DIR / "review"
RESULTS_DIR = DATA_DIR.parent / "results"
SEED = 42

CUSTOMER_RULES = {
    "explicit_human": re.compile(
        r"\b(manager|supervisor|human|real person|someone else|speak (to|with) (a|an|your|the) \w+|talk (to|with) (a|an|your|the) \w+)\b", re.I),
    "anger": re.compile(
        r"\b(furious|angry|ridiculous|unacceptable|worst|terrible|awful|frustrat\w*|annoy\w*|upset|fed up|sick of|"
        r"waste of (my )?time|outrageous|disgust\w*)\b", re.I),
    "repeated_failure": re.compile(
        r"\b(still (not|doesn'?t|didn'?t|won'?t|isn'?t|can'?t|hasn'?t|haven'?t)|already (tried|did|done)|"
        r"tried (that|everything|again)|keeps? (happening|failing)|(not working|failed|happened|broke) again)\b", re.I),
}
AGENT_ESCALATION = re.compile(r"\b(manager|supervisor|escalat\w*|transfer you|specialist)\b", re.I)


def split_for(convo_id) -> str:
    bucket = int(hashlib.md5(str(convo_id).encode()).hexdigest(), 16) % 10
    return "train" if bucket < 8 else "val" if bucket == 8 else "test"


def label_conversation(conv: dict) -> list[dict]:
    turns = [(s, t) for s, t in conv["original"] if s in ("customer", "agent")]
    rows = []
    for i, (speaker, text) in enumerate(turns):
        row = {"conv_id": conv["convo_id"], "turn_idx": i, "speaker": speaker, "text": text,
               "flow": conv["scenario"]["flow"], "subflow": conv["scenario"]["subflow"],
               "escalate": None, "rules": ""}
        if speaker == "customer":
            fired = [name for name, rx in CUSTOMER_RULES.items() if rx.search(text)]
            # the agent's response up to the next customer turn
            for _, agent_text in (t for t in _until_next_customer(turns, i + 1)):
                if AGENT_ESCALATION.search(agent_text):
                    fired.append("agent_escalation")
                    break
            row["escalate"] = int(bool(fired))
            row["rules"] = "|".join(fired)
        rows.append(row)
    return rows


def _until_next_customer(turns, start):
    for speaker, text in turns[start:]:
        if speaker == "customer":
            break
        yield speaker, text


def main() -> None:
    data = json.loads(gzip.decompress(fetch_bytes("abcd_v1.1.json.gz")))
    convs = [c for split in ("train", "dev", "test") for c in data[split]]
    df = pd.DataFrame([row for conv in convs for row in label_conversation(conv)])
    df["split"] = df["conv_id"].map(split_for)

    customer = df[df["speaker"] == "customer"].copy()
    customer["escalate"] = customer["escalate"].astype(int)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"conversations": len(convs), "customer_turns": len(customer), "splits": {}}
    for split, part in customer.groupby("split"):
        part.drop(columns="split").to_csv(PROCESSED_DIR / f"abcd_turns_{split}.csv.gz", index=False)
        stats["splits"][split] = {"conversations": part["conv_id"].nunique(), "turns": len(part),
                                  "positive_turns": int(part["escalate"].sum()),
                                  "conversations_with_positive": int(part.groupby("conv_id")["escalate"].max().sum())}
    stats["rule_counts"] = dict(Counter(r for rules in customer["rules"] if rules for r in rules.split("|")))
    stats["positive_rate"] = float(customer["escalate"].mean())

    # Manual-check export: whole conversations, both speakers, so a reviewer
    # sees the agent turn the agent_escalation rule looked at.
    has_pos = customer.groupby("conv_id")["escalate"].max()
    pos_ids = has_pos[has_pos == 1].sample(n=min(150, int(has_pos.sum())), random_state=SEED).index
    neg_ids = has_pos[has_pos == 0].sample(n=300 - len(pos_ids), random_state=SEED).index
    review = df[df["conv_id"].isin(set(pos_ids) | set(neg_ids))].copy()
    review["escalate"] = review["escalate"].map(lambda v: "" if pd.isna(v) else int(v))
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    review.assign(manual_escalate="", notes="") \
        .rename(columns={"escalate": "auto_escalate"}) \
        .sort_values(["conv_id", "turn_idx"]) \
        .to_csv(REVIEW_DIR / "abcd_review_300.csv", index=False)
    stats["review_export"] = {"conversations": int(review["conv_id"].nunique()), "rows": len(review)}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "abcd_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
