"""End-to-end, side-effect-free evaluation for the Closira pipeline.

Run from the repository root:
    python -m backend.evaluation.evaluate_agent

The pipeline, router and local ML classifier execute unchanged. Groq calls
and database writes are temporarily replaced only for this evaluation process.

test_cases.json still carries the legacy escalation labels in
"expected_intent" (normal / frustrated / explicit_human_request /
repeated_confusion). Those no longer match the model's 7 intents, so they are
reported alongside the predicted intent/emotion/risk but not scored; the
scored checks are escalation, route and stage.
"""

from __future__ import annotations

import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from backend import pipeline


CASES_PATH = Path(__file__).with_name("test_cases.json")
SIDE_EFFECTS = ("llm_call", "llm_extract", "record_escalation", "log_decision", "create_ticket", "save_final_summary")


def deterministic_llm_response(_system: str, _messages: list[dict[str, str]]) -> dict[str, Any]:
    """Offline stand-in for Groq's FAQ reply."""
    return {"response": "I can help with that.", "sop_gap": False}


def no_extraction(_system: str, _messages: list[dict[str, str]]) -> dict[str, Any]:
    """Offline stand-in for Groq slot extraction: extract nothing, so the
    pipeline's deterministic fallbacks fill the slots."""
    return {}


@contextmanager
def offline_dependencies() -> Iterator[None]:
    """Disable only external effects while retaining the production pipeline."""
    originals = {name: getattr(pipeline, name) for name in SIDE_EFFECTS}
    pipeline.llm_call = deterministic_llm_response
    pipeline.llm_extract = no_extraction
    for name in SIDE_EFFECTS[2:]:
        setattr(pipeline, name, lambda *_args, **_kwargs: None)
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(pipeline, name, value)


def load_cases() -> list[dict[str, Any]]:
    with CASES_PATH.open(encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{CASES_PATH} must contain a non-empty JSON array")
    return cases


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    state = pipeline.get_initial_state(session_id=f"evaluation-{case['id']}")
    result = pipeline.run_turn(state, case["message"])
    observed = {
        "actual_intent": result["intent_label"],
        "actual_emotion": result["emotion_label"],
        "actual_risk": result["risk"],
        "actual_escalation": result["stage"] == "escalated",
        "actual_route": "escalation" if result["stage"] == "escalated" else "faq",
        "actual_stage": result["stage"],
        "actual_trigger": result.get("escalation_trigger", ""),
    }
    checks = {
        "escalation": observed["actual_escalation"] == case["expected_escalation"],
        "route": observed["actual_route"] == case["expected_route"],
        "stage": observed["actual_stage"] == case["expected_stage"],
    }
    return {**case, **observed, "checks": checks, "passed": all(checks.values())}


def metric_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [item["expected_escalation"] for item in results]
    actual = [item["actual_escalation"] for item in results]
    precision, recall, f1, _ = precision_recall_fscore_support(expected, actual, pos_label=True, average="binary", zero_division=0)
    return {
        "escalation": {"accuracy": float(accuracy_score(expected, actual)), "precision": float(precision),
                       "recall": float(recall), "f1": float(f1)},
        "routing": {
            "route_accuracy": sum(item["checks"]["route"] for item in results) / len(results),
            "stage_accuracy": sum(item["checks"]["stage"] for item in results) / len(results),
        },
        "triggers": dict(Counter(item["actual_trigger"] or "none" for item in results)),
    }


def print_report(results: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    print(f"Cases: {len(results)} ({dict(Counter(item['category'] for item in results))})")
    e = metrics["escalation"]
    print(f"\nEscalation  accuracy={e['accuracy']:.3f} precision={e['precision']:.3f} recall={e['recall']:.3f} f1={e['f1']:.3f}")
    r = metrics["routing"]
    print(f"Routing     route_accuracy={r['route_accuracy']:.3f} stage_accuracy={r['stage_accuracy']:.3f}")
    print(f"Triggers    {metrics['triggers']}")
    print("\nPer case")
    for item in results:
        mark = "ok  " if item["passed"] else "FAIL"
        print(f"  {mark} {item['id']}: legacy={item['expected_intent']} -> intent={item['actual_intent']} "
              f"emotion={item['actual_emotion']} risk={item['actual_risk']:.2f} stage={item['actual_stage']} "
              f"trigger={item['actual_trigger'] or 'none'}")


def main() -> None:
    cases = load_cases()
    with offline_dependencies():
        results = [run_case(case) for case in cases]
    print_report(results, metric_summary(results))


if __name__ == "__main__":
    main()
