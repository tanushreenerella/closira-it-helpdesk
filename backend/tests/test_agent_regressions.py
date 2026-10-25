"""Regression tests for bugs fixed in the support workflow (now pipeline.py).

These exercise pure state transitions on the pipeline steps directly (no
FastAPI/WebSocket layer, no real Postgres, no real Groq calls, no ML model).
Where a step would normally call Groq, Postgres or the classifier, the call
is monkeypatched so the suite runs fast and offline.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `backend` is importable

from backend import pipeline


def human(text):
    return {"role": "user", "content": text}


def ai(payload):
    return {"role": "assistant", "content": json.dumps(payload)}


def make_state(**overrides):
    """A full AgentState seeded from get_initial_state(), with overrides applied."""
    state = pipeline.get_initial_state("test-session-id")
    state.update(overrides)
    return state


def prediction(intent="information_request", emotion="neutral", intent_conf=0.9, emotion_conf=0.9):
    return {"intent": intent, "intent_confidence": intent_conf, "intent_probs": {intent: intent_conf},
            "emotion": emotion, "emotion_confidence": emotion_conf, "emotion_probs": {emotion: emotion_conf},
            "degraded": False}


@pytest.fixture(autouse=True)
def _no_external_side_effects(monkeypatch):
    """Every test in this file is a pure state-transition test: never touch the
    real database, the decision_log.json file, or the local ML model."""
    monkeypatch.setattr(pipeline, "record_escalation", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "log_decision", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "create_ticket", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "classify_message", lambda text: prediction())
    monkeypatch.setattr(pipeline, "llm_extract", lambda system, messages: {})
    monkeypatch.setattr(pipeline, "llm_select", lambda system, messages: {})


# ── 1. escalation must read the last HUMAN message ──────────────────────────

def test_escalation_uses_last_human_message_not_appended_ai_message():
    """An assistant reply appended after the human message this turn must not
    hide the human message's escalation keywords. Under the old bug the check
    read state["messages"][-1] blindly — the bland assistant text."""
    human_text = "This is unacceptable, I need to speak to a manager right now"
    assistant_text = {
        "response": "Thank you, that helps. I have your employee information, "
                    "issue details, and device or system details. I'll prepare "
                    "a short session summary for Human IT Support.",
        "confidence": 1.0, "escalate": False, "escalation_reason": None,
        "stage_complete": True, "sop_gap": False, "degraded": False,
    }
    state = make_state(
        messages=[human(human_text), ai(assistant_text)],
        intent_label="information_request", intent_confidence=0.0, intent_source_message=human_text,
    )

    state, action, reason, trigger = pipeline.decide(state)
    result = pipeline.escalate_handler(state, reason, trigger)

    assert action == "escalate", "expected escalation triggered by the human message's keywords"
    assert result["stage"] == "escalated"
    assert result["escalation_reason"] == "Angry sentiment detected: 'unacceptable'"


# ── 2. the repeat counter in decide() ───────────────────────────────────────

def test_decide_repeat_count_increments_then_resets_on_route_change():
    """The loop guard tracks whether the SAME route was chosen on consecutive
    turns and resets the moment a different route is chosen. It is keyed on
    the repeated route, not on whether state["qualification"] grew."""
    state = make_state(
        stage="qualify", last_decision="", decision_repeat_count=0, qualification={},
        messages=[human("hello")], intent_label="chit_chat", intent_confidence=0.0, intent_source_message="hello",
    )

    state, action, *_ = pipeline.decide(state)
    assert action == "qualify"
    assert state["last_decision"] == "qualify"
    assert state["decision_repeat_count"] == 1

    # A real answer was recorded between turns, but the guard still
    # increments because the chosen route ("qualify") repeated.
    state["qualification"] = {"answer_1": "Jordan Smith"}
    state, action, *_ = pipeline.decide(state)
    assert state["decision_repeat_count"] == 2

    # Route changes to "faq" -> counter resets to 1.
    state["stage"] = "faq"
    state, action, *_ = pipeline.decide(state)
    assert action == "faq"
    assert state["last_decision"] == "faq"
    assert state["decision_repeat_count"] == 1


def test_decide_forces_escalation_after_repeat_limit():
    state = make_state(stage="faq", last_decision="faq", decision_repeat_count=pipeline.MAX_REPEAT_DECISIONS - 1,
                       messages=[human("how do I connect to wifi")])
    state, action, reason, trigger = pipeline.decide(state)
    assert action == "escalate"
    assert trigger == "turn_limit"


# ── 3. classify() must not trust a stale cached prediction ──────────────────

def test_classify_reclassifies_when_source_message_is_stale(monkeypatch):
    calls = []

    def fake_classify(text):
        calls.append(text)
        return prediction(intent="complaint", emotion="angry", intent_conf=0.95, emotion_conf=0.95)

    monkeypatch.setattr(pipeline, "classify_message", fake_classify)

    current_text = "I want to speak to a human agent please"
    state = make_state(
        messages=[human(current_text)],
        intent_label="chit_chat",   # stale label cached from an earlier turn
        intent_confidence=0.55,
        risk=0.1,
        intent_source_message="an old message from a previous turn",  # does NOT match current_text
    )

    state = pipeline.classify(state)
    state, action, reason, trigger = pipeline.decide(state)

    assert calls == [current_text], "expected a fresh classification of the CURRENT human message"
    assert state["intent_label"] == "complaint"
    assert state["emotion_label"] == "angry"
    assert state["intent_confidence"] == 0.95
    assert action == "escalate"
    assert trigger == "both"  # model risk AND the "human agent" keyword both fire


def test_classify_reuses_cache_when_source_message_matches(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "classify_message", lambda text: calls.append(text) or prediction())

    current_text = "my monitor won't turn on"
    state = make_state(
        messages=[human(current_text)],
        intent_label="issue_report", intent_confidence=0.42, risk=0.2,
        intent_source_message=current_text,  # matches -> reuse cache, do not reclassify
    )

    state = pipeline.classify(state)
    state, action, *_ = pipeline.decide(state)

    assert calls == [], "should not reclassify when the cached prediction already matches the current message"
    assert state["intent_label"] == "issue_report"
    assert state["intent_confidence"] == 0.42
    assert action != "escalate"


# ── 4. filling the last slot stops asking and moves on ──────────────────────
# (was: "qualify stops at 3 answers and hands off to summary" — the fixed
# 3-question script is now slot filling, and a ticket waits until the SOP
# self-help steps have been tried.)

def test_qualify_handler_stops_after_last_slot_and_offers_sop_steps():
    last_answer = "Dell laptop, Windows 11"
    state = make_state(
        stage="qualify",
        qualification_started=True,
        pending_slot="device_os",
        qualification={
            "requester": "Jordan Smith, E1234",
            "issue_category": "vpn",
            "problem_details": "VPN connects and then drops every few minutes",
            "error_message": "error 809",
            "impact": "I can't join client calls",
        },
        messages=[human(last_answer)],
        intent_source_message=last_answer,
    )

    result = pipeline.qualify_handler(state)

    assert result["stage"] == "troubleshoot"
    assert result["stage"] not in ("faq", "qualify")
    assert result["qualification"]["device_os"] == last_answer
    assert all(result["qualification"].get(slot) for slot in pipeline.SLOTS)
    assert result["pending_slot"] == ""
    assert not result.get("ready_for_ticket")

    payload = json.loads(result["messages"][-1]["content"])
    assert "approved VPN client" in payload["response"]
    assert payload["ticket_id"] is None
