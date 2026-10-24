"""End-to-end run_turn() flows with Groq, Postgres and the classifier stubbed."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import pipeline


def prediction(intent="information_request", emotion="neutral", p=0.9):
    return {"intent": intent, "intent_confidence": p, "intent_probs": {intent: p},
            "emotion": emotion, "emotion_confidence": p, "emotion_probs": {emotion: p}, "degraded": False}


@pytest.fixture
def stubs(monkeypatch):
    calls = {"llm": 0, "tickets": [], "summaries": []}

    def llm(system, messages):
        calls["llm"] += 1
        return calls.get("llm_result", {"response": "Restart the VPN client.", "confidence": 0.9, "sop_gap": False,
                                        "escalate": True})  # Groq's "escalate" must be ignored

    monkeypatch.setattr(pipeline, "llm_call", llm)
    monkeypatch.setattr(pipeline, "record_escalation", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "log_decision", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "create_ticket", lambda **k: calls["tickets"].append(k) or {"ticket_id": "T1"})
    monkeypatch.setattr(pipeline, "save_final_summary", lambda sid, s: calls["summaries"].append(s))
    monkeypatch.setattr(pipeline, "generate_summary", lambda state: {"severity": "high", "key_details_collected": {}})
    monkeypatch.setattr(pipeline, "classify_message", lambda text: prediction("issue_report"))
    return calls


def last_reply(state):
    return json.loads(state["messages"][-1]["content"])


def test_full_flow_faq_qualify_summary(stubs):
    state = pipeline.get_initial_state("s1")
    state = pipeline.run_turn(state, "my VPN keeps dropping")
    assert state["stage"] == "qualify"
    assert pipeline.QUALIFY_QUESTIONS[0] in last_reply(state)["response"]
    assert last_reply(state)["escalate"] is False, "Groq's escalate field must not pick the route"
    assert last_reply(state)["intent"] == "issue_report"

    state = pipeline.run_turn(state, "Jordan Smith")
    state = pipeline.run_turn(state, "VPN drops every 10 minutes, blocking client calls")
    assert state["stage"] == "qualify"
    state = pipeline.run_turn(state, "Dell laptop, Windows 11, error 809")

    assert state["stage"] == "summary"
    assert state["conversation_ended"] is True
    assert stubs["llm"] == 1, "Groq is only called for the FAQ reply"
    assert stubs["tickets"][-1]["severity"] == "high"
    assert "Ticket #T1" in last_reply(state)["response"]

    # after handoff, further messages get no reply
    n = len(state["messages"])
    state = pipeline.run_turn(state, "hello?")
    assert len(state["messages"]) == n + 1


def test_greeting_does_not_start_qualification(stubs, monkeypatch):
    monkeypatch.setattr(pipeline, "classify_message", lambda text: prediction("chit_chat"))
    state = pipeline.run_turn(pipeline.get_initial_state("s2"), "good morning there")
    assert state["stage"] == "faq"
    assert state["qualification_started"] is False


def test_sop_gap_escalates_via_router(stubs):
    stubs["llm_result"] = {"response": "I don't have that information right now.", "confidence": 0.4, "sop_gap": True}
    state = pipeline.run_turn(pipeline.get_initial_state("s3"), "what is the office wifi password policy for guests on mars")
    assert state["stage"] == "escalated"
    assert state["escalation_trigger"] == "sop_gap"
    assert state["sop_gaps"] == ["what is the office wifi password policy for guests on mars"]
    assert last_reply(state)["escalate"] is True


def test_model_risk_escalates_and_meta_is_attached(stubs, monkeypatch):
    monkeypatch.setattr(pipeline, "classify_message", lambda text: prediction("complaint", "angry", 0.95))
    state = pipeline.run_turn(pipeline.get_initial_state("s4"), "my laptop died again during a client demo")
    assert state["stage"] == "escalated"
    assert state["escalation_trigger"] == "model"
    reply = last_reply(state)
    assert reply["emotion"] == "angry" and reply["risk"] >= 0.75
    assert stubs["llm"] == 0


def test_classifier_failure_degrades_gracefully(stubs, monkeypatch):
    def boom(text):
        raise RuntimeError("no model")
    monkeypatch.setattr(pipeline, "classify_message", boom)
    state = pipeline.run_turn(pipeline.get_initial_state("s5"), "how do I reset my password")
    assert state["intent_degraded"] is True
    assert state["stage"] == "qualify"
    assert last_reply(state)["classifier_degraded"] is True
