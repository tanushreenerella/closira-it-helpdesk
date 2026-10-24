"""Deterministic routing for Closira.

route() is the ONLY place a next action is chosen. The LLM never picks a
route — it only writes FAQ reply text. Rule flags (anger keywords, explicit
human requests, repeated-failure phrases, SOP gaps, the repeat-loop guard)
are hard overrides that win over anything the ML model says.
"""

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


def keyword_flags(text: str) -> dict:
    """Rule flags for router.route(). Each value is the escalation reason, or ""."""
    return {
        "security_incident": security_check(text)[1],
        "angry_keyword": sentiment_check(text)[1],
        "explicit_human": explicit_escalation(text)[1],
        "repeated_failure": repeated_failure_check(text)[1],
    }


def _expected(probs: dict | None, weights: dict, label: str | None) -> float:
    if probs:
        return sum(weights.get(k, 0.0) * float(p) for k, p in probs.items())
    return weights.get(label or "", 0.0)


def risk_score(intent: str | None = None, emotion: str | None = None,
               intent_probs: dict | None = None, emotion_probs: dict | None = None) -> float:
    """Per-turn escalation risk in [0, 1] from intent + emotion predictions.

    Combined as a noisy-OR: either a hostile emotion or a complaint-like
    intent is enough on its own to push risk up.
    """
    e = _expected(emotion_probs, EMOTION_RISK, emotion)
    i = _expected(intent_probs, INTENT_RISK, intent)
    return round(min(1.0, max(0.0, 1 - (1 - e) * (1 - i))), 4)


def route(intent: str, emotion: str, risk: float, keyword_flags: dict | None = None, *,
          stage: str = "faq", qualification_complete: bool = False) -> tuple[str, str]:
    """Pick the next action. Returns (route, reason).

    Order of precedence:
      1. any hard-override flag          -> escalate
      2. risk over threshold             -> escalate
      3. complaint with elevated risk    -> escalate
      4. qualification complete          -> summary
      5. mid-qualification               -> qualify
      6. everything else                 -> faq
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

    if qualification_complete:
        return "summary", "Qualification complete"
    if stage == "qualify":
        return "qualify", "Collecting issue details"
    return "faq", f"Answer from SOP (intent={intent})"
