"""Unit tests for the deterministic router."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from backend.router import (COMPLAINT_ESCALATE_THRESHOLD, RISK_ESCALATE_THRESHOLD, explicit_escalation, keyword_flags, repeated_failure_check,
                            risk_score, route)


def test_keyword_flags_are_hard_overrides_even_at_zero_risk():
    flags = keyword_flags("can I talk to a real person")
    assert route("information_request", "happy", 0.0, flags) == ("escalate", "Customer requested human agent")


def test_security_keyword_escalates():
    action, reason = route("issue_report", "neutral", 0.1, keyword_flags("I think I clicked a phishing email"))
    assert action == "escalate"
    assert "security" in reason.lower()


def test_sop_gap_flag_escalates():
    assert route("information_request", "neutral", 0.0, {"sop_gap": True})[0] == "escalate"


def test_high_risk_escalates_without_keywords():
    assert route("issue_report", "angry", RISK_ESCALATE_THRESHOLD, {})[0] == "escalate"


def test_complaint_has_lower_escalation_bar():
    between = (COMPLAINT_ESCALATE_THRESHOLD + RISK_ESCALATE_THRESHOLD) / 2
    assert COMPLAINT_ESCALATE_THRESHOLD < RISK_ESCALATE_THRESHOLD
    assert route("complaint", "frustrated", between, {})[0] == "escalate"
    assert route("issue_report", "frustrated", between, {})[0] == "faq"


def test_stage_and_qualification_routing():
    assert route("issue_report", "neutral", 0.1, {}, stage="qualify") == ("qualify", "Collecting issue details")
    assert route("issue_report", "neutral", 0.1, {}, stage="qualify", qualification_complete=True)[0] == "summary"
    assert route("information_request", "neutral", 0.1, {})[0] == "faq"


def test_risk_score_bounds_and_ordering():
    calm = risk_score("information_request", "neutral")
    angry = risk_score(emotion_probs={"angry": 0.9, "neutral": 0.1}, intent_probs={"complaint": 1.0})
    assert 0.0 <= calm < 0.2
    assert angry > RISK_ESCALATE_THRESHOLD
    assert 0.0 <= angry <= 1.0



@pytest.mark.parametrize("text", [
    "Please connect me to a live technician.", "I need a tech agent to unlock it.", "Can someone help me over the phone?",
    "Could you transfer me to someone who can resolve this?", "Please connect me to staff",
    "I need a live chat with IT right now.", "Can I speak to an IT specialist?",
])
def test_expanded_explicit_human_phrases(text):
    assert explicit_escalation(text)[0]


@pytest.mark.parametrize("text", [
    "I followed the reset steps, but I still can’t log in.",  # curly apostrophe
    "It keeps asking for the code.", "I keep getting the same error.", "Nothing works.",
    "That didn't answer my question.", "The instructions didn’t resolve the issue.",
])
def test_expanded_repeated_failure_phrases(text):
    assert repeated_failure_check(text)[0]


@pytest.mark.parametrize("text", [
    "How do I connect to the office Wi-Fi?", "My VPN keeps dropping every 10 minutes.",
    "Can you connect me to the VPN?", "I still need help setting up email on my phone.",
])
def test_expanded_lists_do_not_fire_on_plain_requests(text):
    assert not explicit_escalation(text)[0]
    assert not repeated_failure_check(text)[0]
