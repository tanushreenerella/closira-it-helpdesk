"""Unit tests for the deterministic router."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.router import RISK_ESCALATE_THRESHOLD, keyword_flags, risk_score, route


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
    assert route("complaint", "frustrated", 0.6, {})[0] == "escalate"
    assert route("issue_report", "frustrated", 0.6, {})[0] == "faq"


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
