"""Calibrate the routing thresholds for ~80% escalation recall.

    python -m backend.ml.calibrate [--model multitask|tfidf_logreg|all] [--target-recall 0.8] [--abcd]

Thresholds searched (the same rule the pipeline applies in the FAQ stage):
    intent_threshold     intent confidence below this -> "uncertain"
                         (risk from emotion + emotion keywords only)
    risk_threshold       escalate when risk >= this
    complaint_threshold  escalate a complaint when risk >= this
Keyword hard overrides always escalate and are not tuned.

Data. The only IT-helpdesk messages with escalation labels are the 616
legacy messages (gold = label != "normal"). They are split 50/50,
stratified by legacy label (seed 42): thresholds are chosen on the
calibration half and reported on the held-out half, which stays unseen.
The calibration half is therefore no longer out-of-domain data.

Selection: among configurations reaching the target recall on the
calibration half, take the highest precision (then F1, then the one closest
to the current defaults). If none reaches it, take the highest recall.
Also reported: operating points — the best recall within a false-alarm
budget (share of "normal" messages escalated) of 5/10/20/30%.

--abcd additionally reports recall/precision on the ABCD validation split
(retail chat, rule-derived per-turn labels, ~1% positive) as a cross-domain
sanity check — not used for selection.

Writes results/calibration_<model>.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .inference import available_models, load_predictor, to_prediction
from ..router import (COMPLAINT_ESCALATE_THRESHOLD, HARD_OVERRIDE_FLAGS, INTENT_CONFIDENCE_THRESHOLD,
                      RISK_ESCALATE_THRESHOLD, UNCERTAIN_INTENT, emotion_only_risk, keyword_flags, risk_score, route)

ML_DIR = Path(__file__).resolve().parent
OOD_PATH = ML_DIR / "data" / "ood" / "it_messages_616.csv"
ABCD_VAL = ML_DIR / "data" / "processed" / "abcd_turns_val.csv.gz"
RESULTS_DIR = ML_DIR / "results"
SEED = 42

DEFAULTS = {"intent_threshold": INTENT_CONFIDENCE_THRESHOLD, "risk_threshold": RISK_ESCALATE_THRESHOLD,
            "complaint_threshold": COMPLAINT_ESCALATE_THRESHOLD}
# Operating points: best calibration-half recall within each false-alarm budget
# (share of "normal" messages escalated).
FALSE_ALARM_BUDGETS = [0.05, 0.10, 0.20, 0.30]
GRID = {
    "intent_threshold": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "risk_threshold": [round(float(x), 3) for x in np.arange(0.35, 0.901, 0.025)],
    "complaint_threshold": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
}


def featurize(texts: list[str], model: str) -> list[dict]:
    """Model outputs + keyword flags per text — computed once, reused for every threshold."""
    preds = [to_prediction(p, model) for p in load_predictor(model)(texts)]
    rows = []
    for text, p in zip(texts, preds):
        flags = keyword_flags(text)
        rows.append({
            "p": p, "flags": flags,
            "keyword": any(v for k, v in flags.items() if k in HARD_OVERRIDE_FLAGS),
            # risk is threshold-independent apart from the uncertain switch, so precompute both
            "risk_certain": risk_score(p["intent"], p["emotion"], p["intent_probs"], p["emotion_probs"]),
            "risk_uncertain": emotion_only_risk(p["emotion"], p["emotion_probs"], p["emotion_confidence"], flags),
        })
    return rows


def decide(row: dict, intent_threshold: float, risk_threshold: float, complaint_threshold: float) -> bool:
    p = row["p"]
    certain = p["intent_confidence"] >= intent_threshold
    intent = p["intent"] if certain else UNCERTAIN_INTENT
    risk = row["risk_certain"] if certain else row["risk_uncertain"]
    return route(intent, p["emotion"], risk, row["flags"],
                 risk_threshold=risk_threshold, complaint_threshold=complaint_threshold)[0] == "escalate"


def metrics(rows: list[dict], gold: list[bool], labels: list[str] | None, **cfg) -> dict:
    pred = np.array([decide(r, **cfg) for r in rows])
    gold = np.array(gold)
    tp = int((pred & gold).sum())
    precision = tp / max(1, int(pred.sum()))
    recall = tp / max(1, int(gold.sum()))
    negatives = max(1, int((~gold).sum()))
    out = {"precision": round(precision, 4), "recall": round(recall, 4),
           "f1": round(2 * precision * recall / max(1e-9, precision + recall), 4),
           "false_alarm_rate": round(int((pred & ~gold).sum()) / negatives, 4),
           "escalated": int(pred.sum()), "positives": int(gold.sum())}
    if labels is not None:
        labels = np.array(labels)
        out["escalate_rate_by_legacy_label"] = {str(l): round(float(pred[labels == l].mean()), 4) for l in sorted(set(labels))}
    return out


def distance(cfg):
    return sum(abs(cfg[k] - DEFAULTS[k]) for k in cfg)


def grid_scores(rows, gold) -> list[tuple[dict, dict]]:
    return [(cfg, metrics(rows, gold, None, **cfg))
            for cfg in (dict(zip(GRID, values)) for values in itertools.product(*GRID.values()))]


def operating_point(scored, budget: float) -> dict:
    """Highest recall with false-alarm rate <= budget (then precision, then closest to defaults)."""
    within = [s for s in scored if s[1]["false_alarm_rate"] <= budget]
    return max(within, key=lambda s: (s[1]["recall"], s[1]["precision"], -distance(s[0])))[0]


def select(scored, target: float) -> tuple[dict, dict]:
    reaching = [s for s in scored if s[1]["recall"] >= target]
    if reaching:
        cfg, m = max(reaching, key=lambda s: (s[1]["precision"], s[1]["f1"], -distance(s[0])))
    else:
        cfg, m = max(scored, key=lambda s: (s[1]["recall"], s[1]["precision"], -distance(s[0])))
    return cfg, m


def calibrate(model: str, target: float, abcd: bool) -> dict:
    df = pd.read_csv(OOD_PATH)
    calib, test = train_test_split(df, test_size=0.5, stratify=df["label"], random_state=SEED)
    rows_c, rows_t = featurize(calib["text"].astype(str).tolist(), model), featurize(test["text"].astype(str).tolist(), model)
    gold_c, gold_t = (calib["label"] != "normal").tolist(), (test["label"] != "normal").tolist()

    scored = grid_scores(rows_c, gold_c)
    cfg, calib_m = select(scored, target)
    result = {
        "model": model, "target_recall": target,
        "split": {"calibration": len(calib), "held_out": len(test), "seed": SEED, "stratified_by": "legacy label"},
        "defaults": DEFAULTS, "calibrated": cfg,
        "calibration_half": {"defaults": metrics(rows_c, gold_c, calib["label"].tolist(), **DEFAULTS),
                             "calibrated": metrics(rows_c, gold_c, calib["label"].tolist(), **cfg)},
        "held_out_half": {"defaults": metrics(rows_t, gold_t, test["label"].tolist(), **DEFAULTS),
                          "calibrated": metrics(rows_t, gold_t, test["label"].tolist(), **cfg)},
        "target_reached_on_calibration": calib_m["recall"] >= target,
        "operating_points": [],
    }
    for budget in FALSE_ALARM_BUDGETS:
        op = operating_point(scored, budget)
        result["operating_points"].append({
            "false_alarm_budget": budget, "thresholds": op,
            "calibration_half": metrics(rows_c, gold_c, calib["label"].tolist(), **op),
            "held_out_half": metrics(rows_t, gold_t, test["label"].tolist(), **op),
        })
    if abcd and ABCD_VAL.is_file():
        val = pd.read_csv(ABCD_VAL)
        rows_a = featurize(val["text"].astype(str).tolist(), model)
        gold_a = (val["escalate"] == 1).tolist()
        result["abcd_val_cross_domain"] = {"defaults": metrics(rows_a, gold_a, None, **DEFAULTS),
                                           "calibrated": metrics(rows_a, gold_a, None, **cfg)}
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="all")
    p.add_argument("--target-recall", type=float, default=0.8)
    p.add_argument("--abcd", action="store_true")
    args = p.parse_args(argv)
    models = available_models() if args.model == "all" else [args.model]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for model in models:
        result = calibrate(model, args.target_recall, args.abcd)
        (RESULTS_DIR / f"calibration_{model}.json").write_text(json.dumps(result, indent=2))
        print(f"\n== {model}  calibrated={result['calibrated']}  (target reached: {result['target_reached_on_calibration']})")
        for half in ("calibration_half", "held_out_half", "abcd_val_cross_domain"):
            if half in result:
                for which in ("defaults", "calibrated"):
                    m = result[half][which]
                    print(f"  {half:22s} {which:10s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                          f"FA={m['false_alarm_rate']:.3f} {m.get('escalate_rate_by_legacy_label', '')}")
        for op in result["operating_points"]:
            m = op["held_out_half"]
            print(f"  FA budget {op['false_alarm_budget']:.2f} {op['thresholds']} -> held-out P={m['precision']:.3f} "
                  f"R={m['recall']:.3f} F1={m['f1']:.3f} FA={m['false_alarm_rate']:.3f} {m['escalate_rate_by_legacy_label']}")


if __name__ == "__main__":
    main()
