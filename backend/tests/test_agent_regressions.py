"""Regression tests for bugs fixed in agent.py's LangGraph workflow.

These exercise pure state transitions on the graph nodes directly (no
FastAPI/WebSocket layer, no real Postgres, no real Groq calls). Where a node
would normally call Groq or Postgres, the call is monkeypatched so the suite
runs fast and offline.
"""

import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `backend` is importable

from backend import agent


def make_state(**overrides):
    """A full AgentState seeded from get_initial_state(), with overrides applied."""
    state = agent.get_initial_state("test-session-id")
    state.update(overrides)
    return state


@pytest.fixture(autouse=True)
def _no_external_side_effects(monkeypatch):
    """Every test in this file is a pure state-transition test: never touch the
    real database or the real decision_log.json file on disk, regardless of
    which code path a given test happens to hit."""
    monkeypatch.setattr(agent, "record_escalation", lambda *a, **k: None)
    monkeypatch.setattr(agent, "log_decision", lambda *a, **k: None)
    monkeypatch.setattr(agent, "create_ticket", lambda *a, **k: None)


# ── 1. escalation_result() must read the last HUMAN message ─────────────────

def test_escalation_result_uses_last_human_message_not_appended_ai_message():
    """qualify_node appends its own closing AI reply to state["messages"]
    before handing off to summary_node. escalation_result() (the safety-net
    check summary_node also runs) must still evaluate the last HUMAN message
    for escalation keywords — not that trailing assistant reply.

    Under the old bug, escalation_result() read state["messages"][-1]
    blindly, which after qualify_node's append is the assistant's bland
    "Thank you, that helps..." text (no escalation keywords) — so this
    scenario would have silently failed to escalate.
    """
    human_text = "This is unacceptable, I need to speak to a manager right now"
    assistant_text = json.dumps({
        "response": "Thank you, that helps. I have your employee information, "
                     "issue details, and device or system details. I'll prepare "
                     "a short session summary for Human IT Support.",
        "confidence": 1.0,
        "escalate": False,
        "escalation_reason": None,
        "stage_complete": True,
        "sop_gap": False,
        "degraded": False,
    })
    state = make_state(
        messages=[HumanMessage(content=human_text), AIMessage(content=assistant_text)],
        intent_label="normal",
        intent_confidence=0.0,
        intent_source_message=human_text,
    )

    result = agent.escalation_result(state)

    assert result is not None, "expected escalation triggered by the human message's keywords"
    assert result["stage"] == "escalated"
    assert result["escalation_reason"] == "Angry sentiment detected: 'unacceptable'"


# ── 2. decision_node's repeat counter ────────────────────────────────────────

def test_decision_node_repeat_count_increments_then_resets_on_action_change(monkeypatch):
    """Documents the ACTUAL behavior of decision_node's loop guard: it tracks
    whether the SAME action was chosen on consecutive turns
    (`action == last_decision` in agent.py), and resets the moment a
    different action is chosen.

    Note: this is keyed purely on repeated *action choice*, not on whether
    state["qualification"] grew between turns — the two usually move
    together in the normal 3-question qualify flow, but they are not the
    same signal. Flagged separately outside this test suite.
    """
    actions = iter([
        {"action": "qualify", "reasoning": "need name"},
        {"action": "qualify", "reasoning": "need issue description"},
        {"action": "faq", "reasoning": "user asked something answerable from SOP"},
    ])
    monkeypatch.setattr(agent, "llm_call", lambda *a, **k: next(actions))

    state = make_state(
        stage="qualify",
        last_decision="",
        decision_repeat_count=0,
        qualification={},
        messages=[HumanMessage(content="hello")],
        intent_label="normal",
        intent_confidence=0.0,
        intent_source_message="hello",
    )

    state = agent.decision_node(state)
    assert state["last_decision"] == "qualify"
    assert state["decision_repeat_count"] == 1

    # A real answer was recorded between turns (qualification grew), but the
    # guard still increments because the chosen action ("qualify") repeated.
    state["qualification"] = {"answer_1": "Jordan Smith"}
    state = agent.decision_node(state)
    assert state["decision_repeat_count"] == 2

    # Action changes to "faq" -> counter resets to 1.
    state = agent.decision_node(state)
    assert state["last_decision"] == "faq"
    assert state["decision_repeat_count"] == 1


# ── 3. classifier_escalation() must not trust a stale cached label ──────────

def test_classifier_escalation_reclassifies_when_source_message_is_stale(monkeypatch):
    calls = []

    def fake_classify(text):
        calls.append(text)
        return {"label": "explicit_human_request", "confidence": 0.95}

    monkeypatch.setattr(agent, "classify_employee_message", fake_classify)

    current_text = "I want to speak to a human agent please"
    state = make_state(
        messages=[HumanMessage(content=current_text)],
        intent_label="frustrated",  # stale label cached from an earlier turn
        intent_confidence=0.55,     # stale confidence, below the escalation threshold
        intent_source_message="an old message from a previous turn",  # does NOT match current_text
    )

    should_escalate, reason, response, trigger, label, confidence = agent.classifier_escalation(state)

    assert calls == [current_text], "expected a fresh classification of the CURRENT human message"
    assert label == "explicit_human_request"
    assert confidence == 0.95
    assert should_escalate is True
    assert trigger == "both"  # model AND the "human agent" keyword both fire


def test_classifier_escalation_reuses_cache_when_source_message_matches(monkeypatch):
    calls = []
    monkeypatch.setattr(
        agent, "classify_employee_message",
        lambda text: calls.append(text) or {"label": "normal", "confidence": 0.1},
    )

    current_text = "my monitor won't turn on"
    state = make_state(
        messages=[HumanMessage(content=current_text)],
        intent_label="normal",
        intent_confidence=0.42,
        intent_source_message=current_text,  # matches -> reuse cache, do not reclassify
    )

    should_escalate, reason, response, trigger, label, confidence = agent.classifier_escalation(state)

    assert calls == [], "should not reclassify when the cached label already matches the current message"
    assert label == "normal"
    assert confidence == 0.42
    assert should_escalate is False


# ── 4. qualify_node stops at 3 answers and hands off to summary ─────────────

def test_qualify_node_completes_after_third_answer_and_goes_to_summary():
    third_answer = "Dell laptop, Windows 11, error code 0x80070005"
    state = make_state(
        stage="qualify",
        qualification_started=True,
        qualification={
            "answer_1": "Jordan Smith",
            "answer_2": "VPN keeps disconnecting, blocking my work",
        },
        messages=[HumanMessage(content=third_answer)],
        intent_label="normal",
        intent_confidence=0.0,
        intent_source_message=third_answer,
    )

    result = agent.qualify_node(state)

    assert result["stage"] == "summary"
    assert result["stage"] != "faq"
    assert len(result["qualification"]) == 3
    assert result["qualification"]["answer_3"] == third_answer

    payload = json.loads(result["messages"][-1].content)
    assert payload["stage_complete"] is True
