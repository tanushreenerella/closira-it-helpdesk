"""Deterministic routing for Closira.

route() is the ONLY place a next action is chosen. The LLM never picks a
route — it only writes FAQ reply text and extracts slot values. Rule flags (anger keywords, explicit
human requests, repeated-failure phrases, SOP gaps, the repeat-loop guard)
are hard overrides that win over anything the ML model says.
"""

import re

ROUTES = ("faq", "qualify", "escalate", "summary")

INTENTS = ("information_request", "issue_report", "action_request",
           "complaint", "feedback", "appreciation", "chit_chat")
EMOTIONS = ("angry", "frustrated", "confused", "sad", "happy", "neutral")

# How much each predicted class contributes to escalation risk. Risk is the
# probability-weighted average of these, so a confident "angry" reads close
# to 1.0 and an uncertain one is pulled down by the other classes.
EMOTION_RISK = {"angry": 1.0, "frustrated": 0.75, "confused": 0.35, "sad": 0.4,
                "happy": 0.0, "neutral": 0.05}
INTENT_RISK = {"complaint": 0.8, "issue_report": 0.3, "feedback": 0.2, "action_request": 0.15,
               "information_request": 0.05, "chit_chat": 0.0, "appreciation": 0.0}

RISK_ESCALATE_THRESHOLD = 0.75       # escalate on risk alone
COMPLAINT_ESCALATE_THRESHOLD = 0.5   # lower bar when the intent is a complaint

# Below this intent confidence the intent is reported as "uncertain" and
# ignored by routing and risk (emotion still counts).
INTENT_CONFIDENCE_THRESHOLD = 0.5
UNCERTAIN_INTENT = "uncertain"
# Intent label for a message that answers a slot / troubleshooting question —
# intent classification is skipped for those.
SLOT_ANSWER_INTENT = "slot_answer"

# Intents that are not a support request — answer them, but don't start
# collecting ticket details.
NON_SUPPORT_INTENTS = {"chit_chat", "appreciation", "feedback"}

# Flag name -> escalation reason used when no more specific reason is given.
HARD_OVERRIDE_FLAGS = {
    "security_incident": "Suspected security incident",
    "angry_keyword": "Angry sentiment detected",
    "explicit_human": "Customer requested human agent",
    "repeated_failure": "Repeated failure signal detected",
    "sop_gap": "Question outside SOP data",
    "repeat_limit": "Same action repeated without progress",
}


# ── Keyword rules (hard overrides) ───────────────────────────────────────────
def _find_keyword(text: str, keywords: list[str]) -> str | None:
    text_lower = text.lower()
    return next((kw for kw in keywords if kw in text_lower), None)


def sentiment_check(text: str) -> tuple[bool, str]:
    """Quick sentiment guard — returns (is_negative, reason)."""
    kw = _find_keyword(text, [
        "angry", "furious", "terrible", "horrible", "disgusting", "lawsuit",
        "unacceptable", "scam", "fraud", "worst", "never coming back",
        "this is ridiculous", "i demand", "incompetent", "hate"
    ])
    return (True, f"Angry sentiment detected: '{kw}'") if kw else (False, "")


def explicit_escalation(text: str) -> tuple[bool, str]:
    """Check if user explicitly asks for a human."""
    kw = _find_keyword(text, [
        "speak to a human", "talk to someone", "real person",
        "human", "human agent", "person", "representative", "support agent",
        "manager", "supervisor", "speak to staff", "transfer me to it support"
    ])
    return (True, "Customer requested human agent") if kw else (False, "")


def repeated_failure_check(text: str) -> tuple[bool, str]:
    """Check for repeated troubleshooting attempts that have not resolved the issue."""
    kw = _find_keyword(text, [
        "already tried", "tried again", "tried multiple times",
        "still doesn't work", "still confused", "still do not understand", "nothing is working",
    ])
    return (True, f"Repeated failure signal detected: '{kw}'") if kw else (False, "")


def security_check(text: str) -> tuple[bool, str]:
    """SOP escalation trigger: suspected phishing, malware, or account compromise."""
    kw = _find_keyword(text, [
        "phishing", "malware", "ransomware", "virus", "hacked", "compromised",
        "suspicious link", "clicked a link", "unauthorized access", "someone logged into my account",
    ])
    return (True, f"Suspected security incident: '{kw}'") if kw else (False, "")


TICKET_REQUEST = re.compile(
    r"\b((create|open|raise|log|file|submit|make)\s+(me\s+)?(a\s+|the\s+)?(support\s+)?ticket|just\s+(log|raise|create)\s+it)\b", re.I)


def ticket_request_check(text: str) -> bool:
    """The employee asks for a ticket outright — skip any remaining questions."""
    return bool(TICKET_REQUEST.search(text))


def keyword_flags(text: str) -> dict:
    """Rule flags for router.route(). Escalation flags hold the reason, or "";
    ticket_request is a bool and routes to summary, not escalation."""
    return {
        "security_incident": security_check(text)[1],
        "angry_keyword": sentiment_check(text)[1],
        "explicit_human": explicit_escalation(text)[1],
        "repeated_failure": repeated_failure_check(text)[1],
        "ticket_request": ticket_request_check(text),
    }


def _expected(probs: dict | None, weights: dict, label: str | None) -> float:
    if probs:
        return sum(weights.get(k, 0.0) * float(p) for k, p in probs.items())
    return weights.get(label or "", 0.0)


def risk_score(intent: str | None = None, emotion: str | None = None,
               intent_probs: dict | None = None, emotion_probs: dict | None = None) -> float:
    """Per-turn escalation risk in [0, 1] from intent + emotion predictions.

    Combined as a noisy-OR: either a hostile emotion or a complaint-like
    intent is enough on its own to push risk up. For uncertain intents and
    slot answers use emotion_only_risk() instead.
    """
    e = _expected(emotion_probs, EMOTION_RISK, emotion)
    i = _expected(intent_probs, INTENT_RISK, intent)
    return round(min(1.0, max(0.0, 1 - (1 - e) * (1 - i))), 4)


# A top emotion at or above this confidence contributes its full weight on
# the emotion-only path, so a confident frustrated (0.75) / angry (1.0)
# reaches RISK_ESCALATE_THRESHOLD on its own.
EMOTION_CONFIDENT = 0.6
# Keyword rules that signal emotion, and the risk they carry on the
# emotion-only path. (explicit_human / security already force escalation and
# don't say anything about how upset the employee is.)
KEYWORD_RISK = {"angry_keyword": 1.0, "repeated_failure": 0.75}


def emotion_only_risk(emotion: str | None, emotion_probs: dict | None = None,
                      emotion_confidence: float = 0.0, flags: dict | None = None) -> float:
    """Risk from emotion + keyword rules only — used when the intent is
    uncertain or not classified (slot answers), so a missing intent can't
    dilute a clear emotional signal.

    max(probability-weighted emotion risk,
        full weight of the top emotion if it is confident,
        risk of any emotion keyword rule that fired)
    """
    expected = _expected(emotion_probs, EMOTION_RISK, emotion)
    confident = EMOTION_RISK.get(emotion or "", 0.0) if emotion_confidence >= EMOTION_CONFIDENT else 0.0
    keyword = max((r for name, r in KEYWORD_RISK.items() if (flags or {}).get(name)), default=0.0)
    return round(min(1.0, max(expected, confident, keyword)), 4)


def route(intent: str, emotion: str, risk: float, keyword_flags: dict | None = None, *,
          stage: str = "faq", qualification_complete: bool = False) -> tuple[str, str]:
    """Pick the next action. Returns (route, reason).

    Order of precedence:
      1. any hard-override flag              -> escalate
      2. risk over threshold                 -> escalate
      3. complaint with elevated risk        -> escalate
      4. ticket requested / ready for ticket -> summary
      5. mid-qualification or troubleshooting -> qualify
      6. everything else                     -> faq

    An "uncertain" or "slot_answer" intent never matches rule 3, so it has
    no effect on routing.
    """
    flags = keyword_flags or {}
    for name, default_reason in HARD_OVERRIDE_FLAGS.items():
        value = flags.get(name)
        if value:
            return "escalate", value if isinstance(value, str) else default_reason

    if risk >= RISK_ESCALATE_THRESHOLD:
        return "escalate", f"High escalation risk ({risk:.2f}, emotion={emotion}, intent={intent})"
    if intent == "complaint" and risk >= COMPLAINT_ESCALATE_THRESHOLD:
        return "escalate", f"Complaint with elevated risk ({risk:.2f})"

    if flags.get("ticket_request"):
        return "summary", "Employee asked for a ticket"
    if qualification_complete:
        return "summary", "Qualification complete"
    if stage in ("qualify", "troubleshoot"):
        return "qualify", "Collecting issue details" if stage == "qualify" else "Checking whether the SOP fix worked"
    return "faq", f"Answer from SOP (intent={intent})"


# ── Ticket severity ──────────────────────────────────────────────────────────
SEVERITY_HIGH_RISK = 0.7
SEVERITY_MEDIUM_RISK = 0.4

IMPACT_BLOCKED = re.compile(
    r"\b(can'?t|cannot|unable to) (do )?(any|my|all)( of my)? (work|job)|can'?t work( at all)?|blocked from (all|any|everything)|"
    r"completely (blocked|stuck|down)|totally blocked|(nothing|no work) (gets done|at all)|whole (team|office|department)|"
    r"entire (team|office|department)|everyone|all of us|multiple (people|employees|users)|stopped (all )?work", re.I)
IMPACT_DEGRADED = re.compile(r"\b(slow(er|ing)?|workaround|annoying|some (of my )?work|partly|partially|delay\w*|"
                             r"can still|mostly|intermittent\w*|on and off|sometimes)\b", re.I)
IMPACT_MINOR = re.compile(r"\b(minor|not urgent|no rush|cosmetic|whenever|low priority|just curious|not a big deal)\b", re.I)


def impact_level(impact: str | None) -> str | None:
    """blocked | degraded | minor | None, from the employee's impact answer."""
    if not impact:
        return None
    if IMPACT_BLOCKED.search(impact):
        return "blocked"
    if IMPACT_MINOR.search(impact):
        return "minor"
    if IMPACT_DEGRADED.search(impact):
        return "degraded"
    return None


def ticket_severity(risk: float, impact: str | None = None, level: str | None = None,
                    security: bool = False) -> str:
    """Ticket priority from escalation risk + business impact.

    critical: suspected security incident
    high:     risk >= 0.7, or blocked from all work / several people affected
    medium:   risk >= 0.4, or work slowed / workaround needed
    low:      everything else
    """
    level = level or impact_level(impact)
    if security:
        return "critical"
    if risk >= SEVERITY_HIGH_RISK or level == "blocked":
        return "high"
    if risk >= SEVERITY_MEDIUM_RISK or level == "degraded":
        return "medium"
    return "low"
