"""End-to-end, side-effect-free evaluation for the Closira LangGraph agent.

Run from the repository root:
    python -m backend.evaluation.evaluate_agent

The graph and local DistilBERT classifier execute unchanged. Groq calls and
escalation logging are temporarily replaced only for this evaluation process.
"""

from __future__ import annotations

import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from langchain_core.messages import HumanMessage
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from backend import agent


CASES_PATH = Path(__file__).with_name("test_cases.json")
INTENT_LABELS = ["normal", "frustrated", "explicit_human_request", "repeated_confusion"]


def deterministic_llm_response(_system: str, _messages: list[dict[str, str]]) -> dict[str, Any]:
    """Offline stand-in for Groq; preserves the graph's FAQ-stage transition."""
    return {
        "response": "I can help with that.",
        "confidence": 0.9,
        "escalate": False,
        "escalation_reason": None,
        "stage_complete": False,
        "sop_gap": False,
    }


@contextmanager
def offline_agent_dependencies() -> Iterator[None]:
    """Disable only external effects while retaining the compiled production graph."""
    original_llm_call = agent.llm_call
    original_log_escalation = agent.log_escalation
    agent.llm_call = deterministic_llm_response
    agent.log_escalation = lambda *_args, **_kwargs: None
    try:
        yield
    finally:
        agent.llm_call = original_llm_call
        agent.log_escalation = original_log_escalation


def load_cases() -> list[dict[str, Any]]:
    with CASES_PATH.open(encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{CASES_PATH} must contain a non-empty JSON array")
    return cases


def actual_escalation(state: dict[str, Any]) -> bool:
    if state.get("stage") == "escalated":
        return True
    if not state.get("messages"):
        return False
    message = state["messages"][-1]
    content = message.content if hasattr(message, "content") else message.get("content", "")
    try:
        return bool(json.loads(content).get("escalate", False))
    except (TypeError, json.JSONDecodeError):
        return False


def route_for_stage(stage: str) -> str:
    return "escalation" if stage == "escalated" else "faq"


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    state = agent.get_initial_state(session_id=f"evaluation-{case['id']}")
    state["messages"].append(HumanMessage(content=case["message"]))
    result = agent.APP.invoke(state)
    confidence = float(result["intent_confidence"])
    observed = {
        "actual_intent": result["intent_label"],
        "actual_confidence": confidence,
        "actual_escalation": actual_escalation(result),
        "actual_route": route_for_stage(result["stage"]),
        "actual_stage": result["stage"],
        "actual_trigger": result.get("escalation_trigger", ""),
    }
    checks = {
        "intent": observed["actual_intent"] == case["expected_intent"],
        "escalation": observed["actual_escalation"] == case["expected_escalation"],
        "route": observed["actual_route"] == case["expected_route"],
        "stage": observed["actual_stage"] == case["expected_stage"],
    }
    if "max_expected_confidence" in case:
        checks["confidence"] = confidence <= float(case["max_expected_confidence"])
    return {**case, **observed, "checks": checks, "passed": all(checks.values())}


def percent(value: float) -> str:
    return f"{value:.3f}"


def metric_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    expected_intents = [item["expected_intent"] for item in results]
    actual_intents = [item["actual_intent"] for item in results]
    precision, recall, f1, support = precision_recall_fscore_support(
        expected_intents, actual_intents, labels=INTENT_LABELS, zero_division=0
    )
    intent_per_label = {
        label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for label, p, r, f, s in zip(INTENT_LABELS, precision, recall, f1, support)
    }
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        expected_intents, actual_intents, labels=INTENT_LABELS, average="macro", zero_division=0
    )
    expected_escalations = [item["expected_escalation"] for item in results]
    actual_escalations = [item["actual_escalation"] for item in results]
    escalation_precision, escalation_recall, escalation_f1, _ = precision_recall_fscore_support(
        expected_escalations, actual_escalations, pos_label=True, average="binary", zero_division=0
    )
    return {
        "intent": {
            "accuracy": float(accuracy_score(expected_intents, actual_intents)),
            "precision_macro": float(macro_precision),
            "recall_macro": float(macro_recall),
            "f1_macro": float(macro_f1),
            "per_label": intent_per_label,
        },
        "escalation": {
            "accuracy": float(accuracy_score(expected_escalations, actual_escalations)),
            "precision": float(escalation_precision),
            "recall": float(escalation_recall),
            "f1": float(escalation_f1),
        },
        "routing": {
            "route_accuracy": sum(item["checks"]["route"] for item in results) / len(results),
            "stage_accuracy": sum(item["checks"]["stage"] for item in results) / len(results),
        },
    }


def print_report(results: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    print(f"Cases: {len(results)} ({dict(Counter(item['category'] for item in results))})")
    intent = metrics["intent"]
    print("\nIntent metrics")
    print(f"  accuracy={percent(intent['accuracy'])} precision_macro={percent(intent['precision_macro'])} "
          f"recall_macro={percent(intent['recall_macro'])} f1_macro={percent(intent['f1_macro'])}")
    for label, values in intent["per_label"].items():
        print(f"  {label}: precision={percent(values['precision'])} recall={percent(values['recall'])} "
              f"f1={percent(values['f1'])} support={values['support']}")
    escalation = metrics["escalation"]
    print("\nEscalation metrics")
    print(f"  accuracy={percent(escalation['accuracy'])} precision={percent(escalation['precision'])} "
          f"recall={percent(escalation['recall'])} f1={percent(escalation['f1'])}")
    routing = metrics["routing"]
    print("\nRouting metrics")
    print(f"  route_accuracy={percent(routing['route_accuracy'])} stage_accuracy={percent(routing['stage_accuracy'])}")
    failed = [item for item in results if not item["passed"]]
    print(f"\nFailed cases: {len(failed)}")
    for item in failed:
        failed_checks = ", ".join(name for name, passed in item["checks"].items() if not passed)
        print(f"  {item['id']}: failed={failed_checks}; intent {item['expected_intent']} -> "
              f"{item['actual_intent']} ({item['actual_confidence']:.3f}); escalation "
              f"{item['expected_escalation']} -> {item['actual_escalation']}; route/stage "
              f"{item['expected_route']}/{item['expected_stage']} -> "
              f"{item['actual_route']}/{item['actual_stage']}; trigger={item['actual_trigger'] or 'none'}")


def main() -> None:
    cases = load_cases()
    with offline_agent_dependencies():
        results = [run_case(case) for case in cases]
    print_report(results, metric_summary(results))


if __name__ == "__main__":
    main()
