"""End-to-end run_turn() flows with Groq, Postgres and the classifier stubbed."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import pipeline


def prediction(intent="issue_report", emotion="neutral", p=0.9, emotion_p=None):
    emotion_p = p if emotion_p is None else emotion_p
    return {"intent": intent, "intent_confidence": p, "intent_probs": {intent: p},
            "emotion": emotion, "emotion_confidence": emotion_p, "emotion_probs": {emotion: emotion_p}, "degraded": False}


@pytest.fixture
def stubs(monkeypatch):
    """Groq, DB and classifier stubs. Tests queue slot extractions in
    calls["extract"] — one dict per extraction call, {} once the queue is empty."""
    calls = {"llm": 0, "extract": [], "extract_calls": 0, "tickets": [], "summaries": [], "classified": []}

    def llm(system, messages):
        calls["llm"] += 1
        return calls.get("llm_result", {"response": "Restart the VPN client.", "confidence": 0.9, "sop_gap": False,
                                        "escalate": True})  # Groq's "escalate" must be ignored

    def extract(system, messages):
        calls["extract_calls"] += 1
        return calls["extract"].pop(0) if calls["extract"] else {}

    def classify(text):
        calls["classified"].append(text)
        return calls.get("prediction", prediction())

    monkeypatch.setattr(pipeline, "llm_call", llm)
    monkeypatch.setattr(pipeline, "llm_extract", extract)
    monkeypatch.setattr(pipeline, "record_escalation", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "log_decision", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "create_ticket", lambda **k: calls["tickets"].append(k) or {"ticket_id": "T1"})
    monkeypatch.setattr(pipeline, "save_final_summary", lambda sid, s: calls["summaries"].append(s))
    monkeypatch.setattr(pipeline, "generate_summary", lambda state, outcome="": {"key_details_collected": {}})
    monkeypatch.setattr(pipeline, "classify_message", classify)
    return calls


def last_reply(state):
    return json.loads(state["messages"][-1]["content"])


def replies(state):
    return [json.loads(m["content"]) for m in state["messages"] if m["role"] == "assistant"]


REQ = {"requester": "Jordan Smith, E1234"}  # first-message extractions include the optional requester
VPN_FULL = {**REQ, "issue_category": "vpn", "problem_details": "VPN connects then drops every 10 minutes",
            "error_message": "error 809", "device_os": "Dell laptop, Windows 11"}


# ── Slot filling ─────────────────────────────────────────────────────────────

def test_vague_issue_gets_a_category_specific_follow_up(stubs):
    stubs["extract"] = [{**REQ, "issue_category": "vpn", "problem_details": "my vpn is not working properly"}]
    state = pipeline.run_turn(pipeline.get_initial_state("v1"), "my vpn is not working properly")

    assert state["stage"] == "qualify"
    assert state["pending_slot"] == "problem_details"
    assert pipeline.VAGUE_FOLLOWUPS["vpn"] in last_reply(state)["response"]

    # the follow-up answer is added to the details, and the flow moves on
    stubs["extract"] = [{"problem_details": "it connects and then drops after a few minutes"}]
    state = pipeline.run_turn(state, "it connects and then drops after a few minutes")
    assert "drops after a few minutes" in state["qualification"]["problem_details"]
    assert "not working properly" in state["qualification"]["problem_details"]
    assert state["pending_slot"] == "error_message"


def test_vague_follow_up_is_asked_only_once(stubs):
    stubs["extract"] = [{**REQ, "issue_category": "device", "problem_details": "laptop broken"}]
    state = pipeline.run_turn(pipeline.get_initial_state("v2"), "laptop broken")
    assert state["pending_slot"] == "problem_details"
    state = pipeline.run_turn(state, "it's just broken")
    assert state["pending_slot"] != "problem_details"


def test_specific_issue_gets_no_follow_up(stubs):
    stubs["extract"] = [{**REQ, "issue_category": "vpn", "problem_details": "VPN client shows error 809 when connecting"}]
    state = pipeline.run_turn(pipeline.get_initial_state("v3"), "VPN client shows error 809 when connecting")
    assert state["pending_slot"] != "problem_details"


def test_already_given_slots_are_not_re_asked(stubs):
    # first message states category, details, error and device in one go
    stubs["extract"] = [dict(VPN_FULL)]
    state = pipeline.run_turn(pipeline.get_initial_state("s1"),
                              "VPN on my Dell Windows 11 laptop connects then drops every 10 minutes, error 809")
    assert state["pending_slot"] == "impact", "only the missing slot should be asked"
    asked = " ".join(r["response"] for r in replies(state))
    for slot in ("requester", "issue_category", "error_message", "device_os"):
        assert pipeline.SLOT_QUESTIONS[slot].get("vpn", pipeline.SLOT_QUESTIONS[slot]["default"]) not in asked


def test_slot_given_out_of_order_is_not_asked_later(stubs):
    stubs["extract"] = [{**REQ, "issue_category": "email", "problem_details": "emails to clients bounce back"}]
    state = pipeline.run_turn(pipeline.get_initial_state("s2"), "emails to clients bounce back")
    assert state["pending_slot"] == "error_message"

    # answering the error question but also volunteering the device
    stubs["extract"] = [{"error_message": "550 mailbox unavailable", "device_os": "Outlook on a MacBook, macOS 14"}]
    state = pipeline.run_turn(state, "550 mailbox unavailable, I'm on Outlook on my MacBook")
    assert state["qualification"]["device_os"] == "Outlook on a MacBook, macOS 14"
    assert state["pending_slot"] == "impact"


def test_questions_are_phrased_for_the_issue_category(stubs):
    stubs["extract"] = [{**REQ, "issue_category": "network", "problem_details": "Wi-Fi drops every few minutes in the office"}]
    state = pipeline.run_turn(pipeline.get_initial_state("s3"), "Wi-Fi drops every few minutes in the office")
    assert pipeline.SLOT_QUESTIONS["error_message"]["network"] in last_reply(state)["response"]


def test_extraction_failure_falls_back_to_the_asked_slot(stubs, monkeypatch):
    def boom(system, messages):
        raise RuntimeError("groq down")
    monkeypatch.setattr(pipeline, "llm_extract", boom)
    state = pipeline.run_turn(pipeline.get_initial_state("s4"), "my VPN connection drops after 10 minutes")
    assert state["qualification"]["issue_category"] == "vpn"  # keyword fallback
    assert state["pending_slot"] == "requester"
    state = pipeline.run_turn(state, "skip")
    assert state["pending_slot"] == "error_message"
    state = pipeline.run_turn(state, "error 809")
    assert state["qualification"]["error_message"] == "error 809"


# ── SOP troubleshooting before any ticket ───────────────────────────────────

def fill_all_slots(stubs, session="t"):
    stubs["extract"] = [dict(VPN_FULL)]
    state = pipeline.run_turn(pipeline.get_initial_state(session), "VPN drops every 10 minutes, error 809, Windows 11 Dell")
    stubs["extract"] = [{"impact": "slows me down a bit", "impact_level": "degraded"}]
    return pipeline.run_turn(state, "slows me down a bit")


def test_sop_fix_is_offered_before_any_ticket(stubs):
    state = fill_all_slots(stubs)
    assert state["stage"] == "troubleshoot"
    assert stubs["tickets"] == [], "no ticket before the SOP steps have been tried"
    text = last_reply(state)["response"]
    for step in pipeline.PLAYBOOKS["vpn"]["self_help_steps"]:
        assert step in text
    assert "did that fix" in text.lower()


def test_resolved_by_sop_fix_creates_no_ticket(stubs):
    state = pipeline.run_turn(fill_all_slots(stubs), "yes, that worked!")
    assert state["conversation_ended"] is True
    assert stubs["tickets"] == []
    assert last_reply(state)["resolved"] is True
    assert stubs["summaries"][-1]["resolved"] is True


def test_unresolved_after_sop_fix_creates_ticket(stubs):
    state = pipeline.run_turn(fill_all_slots(stubs), "no, same problem")
    assert state["conversation_ended"] is True
    assert len(stubs["tickets"]) == 1
    assert "Ticket #T1" in last_reply(state)["response"]
    assert stubs["tickets"][0]["device_system"] == "Dell laptop, Windows 11; Error: error 809"


def test_unclear_fix_answer_is_reasked_once_then_ticketed(stubs):
    state = pipeline.run_turn(fill_all_slots(stubs), "hmm")
    assert state["stage"] == "troubleshoot" and stubs["tickets"] == []
    state = pipeline.run_turn(state, "hmm")
    assert len(stubs["tickets"]) == 1


def test_ticket_on_request_skips_remaining_questions(stubs):
    stubs["extract"] = [{**REQ, "issue_category": "vpn", "problem_details": "VPN fails with error 809"}]
    state = pipeline.run_turn(pipeline.get_initial_state("t2"), "VPN fails with error 809")
    assert state["stage"] == "qualify"
    state = pipeline.run_turn(state, "please just create a ticket")
    assert state["conversation_ended"] is True
    assert len(stubs["tickets"]) == 1


# ── Severity from risk + impact ──────────────────────────────────────────────

def run_to_ticket(stubs, impact, level, emotion="neutral", emotion_p=0.9):
    stubs["prediction"] = prediction("issue_report", emotion, p=0.9, emotion_p=emotion_p)
    stubs["extract"] = [dict(VPN_FULL)]
    state = pipeline.run_turn(pipeline.get_initial_state("sev"), "VPN drops every 10 minutes, error 809, Windows 11 Dell")
    stubs["prediction"] = prediction("issue_report", "neutral")  # calm slot answers afterwards
    stubs["extract"] = [{"impact": impact, "impact_level": level}]
    state = pipeline.run_turn(state, impact)
    return pipeline.run_turn(state, "no")


def test_severity_high_when_risk_is_high(stubs):
    # first message: frustrated 0.85 x issue_report 0.9 -> risk ~0.735, >= 0.7 but below escalation
    state = run_to_ticket(stubs, "it slows me down", "degraded", emotion="frustrated", emotion_p=0.85)
    assert 0.7 <= state["peak_risk"] < pipeline.RISK_ESCALATE_THRESHOLD
    assert stubs["tickets"][-1]["severity"] == "high"


def test_severity_high_when_blocked_from_all_work(stubs):
    state = run_to_ticket(stubs, "I can't do any work at all", "blocked")
    assert state["peak_risk"] < 0.7
    assert stubs["tickets"][-1]["severity"] == "high"


def test_severity_low_when_risk_low_and_impact_minor(stubs):
    stubs_pred = prediction("information_request", "happy")
    stubs["prediction"] = stubs_pred
    stubs["extract"] = [dict(VPN_FULL)]
    state = pipeline.run_turn(pipeline.get_initial_state("sev2"), "quick one: VPN drops every 10 minutes, error 809")
    stubs["extract"] = [{"impact": "minor, no rush", "impact_level": "minor"}]
    state = pipeline.run_turn(state, "minor, no rush")
    state = pipeline.run_turn(state, "no")
    assert state["peak_risk"] < 0.4
    assert stubs["tickets"][-1]["severity"] == "low"


def test_severity_uses_peak_risk_not_last_turn(stubs):
    stubs["prediction"] = prediction("issue_report", "frustrated", emotion_p=0.85)
    stubs["extract"] = [dict(VPN_FULL)]
    state = pipeline.run_turn(pipeline.get_initial_state("sev3"), "VPN drops every 10 minutes again, error 809")
    stubs["prediction"] = prediction("issue_report", "neutral")  # calm slot answers afterwards
    stubs["extract"] = [{"impact": "slows me down", "impact_level": "degraded"}]
    state = pipeline.run_turn(state, "slows me down")
    state = pipeline.run_turn(state, "no")
    assert state["risk"] < 0.4 <= 0.7 <= state["peak_risk"]
    assert stubs["tickets"][-1]["severity"] == "high"


# ── Classification rules ─────────────────────────────────────────────────────

def test_slot_answers_skip_intent_classification(stubs):
    stubs["extract"] = [{**REQ, "issue_category": "vpn", "problem_details": "VPN fails with error 809 on connect"}]
    state = pipeline.run_turn(pipeline.get_initial_state("c1"), "VPN fails with error 809 on connect")
    stubs["prediction"] = prediction("complaint", "neutral", p=0.99, emotion_p=0.9)
    state = pipeline.run_turn(state, "Windows 11")
    assert state["intent_label"] == "slot_answer"
    assert state["stage"] == "qualify", "a 'complaint' intent on a slot answer must not drive routing"


def test_low_confidence_intent_is_uncertain_and_ignored(stubs):
    stubs["prediction"] = prediction("complaint", "neutral", p=0.45, emotion_p=0.9)
    state = pipeline.classify({**pipeline.get_initial_state("c2"), "messages": [{"role": "user", "content": "the thing"}]})
    assert state["intent_label"] == "uncertain"
    assert state["intent_raw"] == "complaint"
    # risk from emotion alone (neutral), not from the 0.8 complaint weight
    assert state["risk"] < 0.1


def test_reply_confidence_is_not_in_meta(stubs):
    state = pipeline.run_turn(pipeline.get_initial_state("c3"), "my VPN keeps dropping")
    assert "confidence" not in last_reply(state)
    assert last_reply(state)["escalate"] is False, "Groq's escalate field must not pick the route"
    assert stubs["llm"] == 1


# ── Other flows ──────────────────────────────────────────────────────────────

def test_greeting_does_not_start_qualification(stubs):
    stubs["prediction"] = prediction("chit_chat")
    state = pipeline.run_turn(pipeline.get_initial_state("o1"), "good morning there")
    assert state["stage"] == "faq"
    assert state["qualification_started"] is False


def test_sop_gap_escalates_via_router(stubs):
    stubs["llm_result"] = {"response": "I don't have that information right now.", "sop_gap": True}
    state = pipeline.run_turn(pipeline.get_initial_state("o2"), "what is the guest policy for the mars office")
    assert state["stage"] == "escalated"
    assert state["escalation_trigger"] == "sop_gap"
    assert state["sop_gaps"] == ["what is the guest policy for the mars office"]


def test_model_risk_escalates_with_risk_based_severity(stubs):
    stubs["prediction"] = prediction("complaint", "angry", 0.95)
    state = pipeline.run_turn(pipeline.get_initial_state("o3"), "my laptop died again during a client demo")
    assert state["stage"] == "escalated"
    assert state["escalation_trigger"] == "model"
    assert last_reply(state)["ticket_severity"] == "high"
    assert stubs["llm"] == 0


def test_security_escalation_is_critical(stubs):
    state = pipeline.run_turn(pipeline.get_initial_state("o4"), "I clicked a link in a phishing email and typed my password")
    assert state["stage"] == "escalated"
    assert stubs["tickets"][-1]["severity"] == "critical"


def test_classifier_failure_degrades_gracefully(stubs, monkeypatch):
    def boom(text):
        raise RuntimeError("no model")
    monkeypatch.setattr(pipeline, "classify_message", boom)
    state = pipeline.run_turn(pipeline.get_initial_state("o5"), "how do I reset my password")
    assert state["intent_degraded"] is True
    assert state["intent_label"] == "uncertain"
    assert state["stage"] == "qualify"
    assert last_reply(state)["classifier_degraded"] is True


def test_long_slot_filling_does_not_trip_the_repeat_guard(stubs):
    stubs["extract"] = [{**REQ, "issue_category": "device", "problem_details": "laptop broken"}]
    state = pipeline.run_turn(pipeline.get_initial_state("o6"), "laptop broken")
    for answer in ("it won't power on after the update", "none", "Lenovo T14, Windows 11", "I can't do any work at all"):
        state = pipeline.run_turn(state, answer)
    assert state["stage"] == "troubleshoot", state["escalation_reason"]


# ── Optional requester slot ──────────────────────────────────────────────────

def test_requester_is_asked_first_and_stored_on_ticket(stubs):
    stubs["extract"] = [{"issue_category": "vpn", "problem_details": "VPN fails with error 809 on connect"}]
    state = pipeline.run_turn(pipeline.get_initial_state("r1"), "VPN fails with error 809 on connect")
    assert state["pending_slot"] == "requester"
    assert pipeline.SLOT_QUESTIONS["requester"]["default"] in last_reply(state)["response"]

    stubs["extract"] = [{"requester": "Priya Nair, E4471"}]
    state = pipeline.run_turn(state, "Priya Nair, E4471")
    assert state["qualification"]["requester"] == "Priya Nair, E4471"
    state = pipeline.run_turn(state, "create a ticket please")
    assert stubs["tickets"][-1]["employee"] == "Priya Nair, E4471"


def test_requester_can_be_skipped(stubs):
    stubs["extract"] = [{"issue_category": "vpn", "problem_details": "VPN fails with error 809 on connect"}]
    state = pipeline.run_turn(pipeline.get_initial_state("r2"), "VPN fails with error 809 on connect")
    state = pipeline.run_turn(state, "I'd rather not say")
    assert state["qualification"]["requester"] == pipeline.REQUESTER_SKIPPED
    assert state["pending_slot"] != "requester", "a skipped requester is not asked again"
    state = pipeline.run_turn(state, "create a ticket please")
    assert stubs["tickets"][-1]["employee"] is None


def test_requester_given_up_front_is_not_asked(stubs):
    stubs["extract"] = [{"requester": "Sam Lee", "issue_category": "email", "problem_details": "emails bounce with 550"}]
    state = pipeline.run_turn(pipeline.get_initial_state("r3"), "Hi, Sam Lee here — my emails bounce with 550")
    assert state["pending_slot"] != "requester"


# ── Emotion-only risk for uncertain intents ──────────────────────────────────

def test_confident_frustration_escalates_alone_when_intent_uncertain(stubs):
    stubs["prediction"] = prediction("issue_report", "frustrated", p=0.4, emotion_p=0.8)
    state = pipeline.run_turn(pipeline.get_initial_state("u1"), "this wifi thing again, seriously")
    assert state["intent_label"] == "uncertain"
    assert state["risk"] >= pipeline.RISK_ESCALATE_THRESHOLD
    assert state["stage"] == "escalated"
    assert state["escalation_trigger"] == "model"


def test_unconfident_emotion_with_uncertain_intent_does_not_escalate(stubs):
    stubs["prediction"] = prediction("issue_report", "frustrated", p=0.4, emotion_p=0.5)
    state = pipeline.run_turn(pipeline.get_initial_state("u2"), "wifi again")
    assert state["intent_label"] == "uncertain"
    assert state["stage"] != "escalated"
