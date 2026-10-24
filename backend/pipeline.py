"""
Closira AI IT Helpdesk Support Agent — plain-Python turn pipeline.

    run_turn(state, msg): classify -> route -> handler

Classification is the local ML model (backend/ml), routing is the
deterministic backend/router.py, and Groq is only used to write the FAQ reply
text and the end-of-session summary. Groq never chooses a route.

Messages are plain dicts: {"role": "user", "content": str} and
{"role": "assistant", "content": <JSON string of the reply payload>} — the
JSON-string convention is what database.py persists and restores.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import TypedDict, Literal

from groq import Groq
from dotenv import load_dotenv
try:
    from .ml.inference import classify_message
    from .router import (route, risk_score, keyword_flags, sentiment_check, explicit_escalation, repeated_failure_check,
                         security_check, NON_SUPPORT_INTENTS, RISK_ESCALATE_THRESHOLD, COMPLAINT_ESCALATE_THRESHOLD)
    from .database import record_escalation, record_decision, save_final_summary, create_ticket
except ImportError:
    from ml.inference import classify_message
    from router import (route, risk_score, keyword_flags, sentiment_check, explicit_escalation, repeated_failure_check,
                        security_check, NON_SUPPORT_INTENTS, RISK_ESCALATE_THRESHOLD, COMPLAINT_ESCALATE_THRESHOLD)
    from database import record_escalation, record_decision, save_final_summary, create_ticket

load_dotenv()

# ── Load SOP ────────────────────────────────────────────────────────────────
with open(os.path.join(os.path.dirname(__file__), "sop.json"), "r") as f:
    SOP = json.load(f)

SOP_TEXT = json.dumps(SOP, indent=2)

# ── Groq client ──────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"

# ── Config ───────────────────────────────────────────────────────────────────
TRIVIAL_MESSAGES = {"hi", "hello", "hey", "hiya", "ok", "okay", "thanks", "thank you", "yes", "no", "yep", "nope"}
GREETINGS = {"hi", "hello", "hey", "hiya"}
# Escalations happen before the summary's Groq call ever classifies severity,
# so default to the same neutral middle value the summary fallback uses.
DEFAULT_ESCALATION_SEVERITY = "medium"
MAX_REPEAT_DECISIONS = 5  # loop guard: force escalation after this many identical non-terminal routes in a row

DECISION_LOG_PATH = os.path.join(os.path.dirname(__file__), "decision_log.json")


def log_escalation(session_id: str, reason: str, trigger: str = "model",
                   intent_label: str = "", intent_confidence: float = 0.0):
    record_escalation(session_id, reason, trigger, intent_label, intent_confidence)


def log_decision(session_id: str, chosen: str, overridden_by: str, reason: str):
    """Persist a routing decision to Postgres (routing_decisions table).

    Also appends to the local decision_log.json file for backward
    compatibility — that file doesn't survive a deployed host with an
    ephemeral filesystem, so Postgres is the durable copy. Neither write may
    ever break the turn.
    """
    try:
        record_decision(session_id, chosen, overridden_by, reason)
    except Exception:
        pass

    entry = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chosen_action": chosen,
        "overridden_by": overridden_by,  # "" if not overridden, else "keyword"/"model"/"both"/"sop_gap"/"turn_limit"
        "reason": reason,
    }
    try:
        if os.path.exists(DECISION_LOG_PATH):
            with open(DECISION_LOG_PATH, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(entry)
        with open(DECISION_LOG_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


# ── State ────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: list
    session_id: str
    stage: Literal["faq", "qualify", "escalated", "summary"]
    unanswered_count: int
    qualification: dict
    qualification_started: bool
    escalation_reason: str
    sop_gaps: list
    conversation_ended: bool
    intent_label: str
    intent_confidence: float
    emotion_label: str
    emotion_confidence: float
    risk: float
    intent_source_message: str
    intent_degraded: bool
    escalation_trigger: str
    last_decision: str
    decision_repeat_count: int


# ── Prompts ──────────────────────────────────────────────────────────────────
SYSTEM_BASE = f"""You are the AI Support Assistant for Closira IT Helpdesk, providing employee IT support.

YOUR STRICT RULES:
1. Answer ONLY from the SOP data below. Do NOT invent systems, access, policies, troubleshooting results, or any facts.
2. If a question is outside the SOP, say clearly: "I don't have that information right now", offer to connect them with Human IT Support, and set sop_gap=true.
3. Never ask for passwords, one-time codes, recovery codes, or sensitive email content.
4. Be concise, practical, and professional — like a knowledgeable IT support analyst.
5. Always respond in English.

SOP DATA:
{SOP_TEXT}

RESPONSE FORMAT (always return valid JSON, nothing else, no markdown fences):
{{
  "response": "<your reply to the employee>",
  "confidence": <0.0-1.0 float>,
  "sop_gap": <true/false>
}}
"""

QUALIFY_QUESTIONS = [
    "What is your name or employee identifier, if you are comfortable sharing it?",
    "Please describe the issue and the impact it is having on your work.",
    "What device and operating system are you using, and what exact error message do you see?"
]

ESCALATION_RESPONSES = {
    "upset": "I'm sorry this has been frustrating. I'm going to connect you with one of our team members right away.",
    "human": "I'm flagging this conversation for Human IT Support right away. An IT support analyst will follow up with you shortly.",
    "repeated": "I can see the previous guidance has not resolved this. I'm escalating this to Human IT Support so they can take over.",
    "security": "This may be a security issue, so I'm escalating it to Human IT Support immediately. Please don't enter any more credentials until they contact you.",
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def user_message(content: str) -> dict:
    return {"role": "user", "content": content}


def assistant_message(payload: dict) -> dict:
    return {"role": "assistant", "content": json.dumps(payload)}


def llm_call(system: str, messages: list) -> dict:
    """Call Groq for FAQ reply text and parse its JSON. Only response,
    confidence and sop_gap are read from the result."""
    groq_messages = [{"role": "system", "content": system}] + messages
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=groq_messages
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        # Clean up response field if Groq appended a JSON blob inside it
        resp = str(parsed.get("response", ""))
        if "{" in resp:
            parsed["response"] = resp[:resp.index("{")].strip()
        return parsed
    except json.JSONDecodeError:
        return {"response": raw, "confidence": 0.5, "sop_gap": False}


def assistant_payload(result: dict) -> dict:
    """Keep assistant messages structured so the UI can read confidence/routing data."""
    return {
        "response": str(result.get("response", "")).strip(),
        "confidence": float(result.get("confidence", 0.7) or 0.7),
        "escalate": bool(result.get("escalate", False)),
        "escalation_reason": result.get("escalation_reason"),
        "stage_complete": bool(result.get("stage_complete", False)),
        "sop_gap": bool(result.get("sop_gap", False)),
        "degraded": bool(result.get("degraded", False)),
        "ticket_id": result.get("ticket_id"),
        "ticket_severity": result.get("ticket_severity"),
    }


def last_human_message(state: AgentState) -> str | None:
    """Content of the most recent user message, regardless of any assistant
    replies appended after it this turn."""
    for m in reversed(state["messages"]):
        if m.get("role") == "user":
            return m.get("content", "")
    return None


def format_messages(state: AgentState) -> list:
    """Convert state messages to Groq chat format."""
    formatted = []
    for m in state["messages"]:
        role, content = m.get("role"), m.get("content", "")
        if role == "assistant":
            try:
                content = json.loads(content).get("response", content)
            except Exception:
                pass
            formatted.append({"role": "assistant", "content": content})
        elif role == "user":
            formatted.append({"role": "user", "content": content})
    return formatted


# ── Step 1: classify ─────────────────────────────────────────────────────────
def neutral_prediction(degraded: bool = False, intent: str = "information_request") -> dict:
    return {"intent": intent, "intent_confidence": 0.0, "intent_probs": None,
            "emotion": "neutral", "emotion_confidence": 0.0, "emotion_probs": None,
            "degraded": degraded}


def classify_text(text: str) -> dict:
    if text.lower().strip() in TRIVIAL_MESSAGES:
        return neutral_prediction(intent="chit_chat")
    try:
        return classify_message(text)
    except Exception as exc:
        print(f"[classifier] Failed to classify: {exc}")
        return neutral_prediction(degraded=True)


def classify(state: AgentState) -> AgentState:
    """Attach intent / emotion / risk for the latest user message to state.

    Re-uses the cached prediction only when it was computed for THIS message
    (tracked via intent_source_message) — never a label left over from an
    earlier turn.
    """
    text = last_human_message(state)
    if text is None or state.get("intent_source_message") == text:
        return state
    p = classify_text(text)
    return {
        **state,
        "intent_label": p["intent"], "intent_confidence": p["intent_confidence"],
        "emotion_label": p["emotion"], "emotion_confidence": p["emotion_confidence"],
        "risk": risk_score(p["intent"], p["emotion"], p.get("intent_probs"), p.get("emotion_probs")),
        "intent_source_message": text, "intent_degraded": p.get("degraded", False),
    }


# ── Step 2: route ────────────────────────────────────────────────────────────
def model_escalates(state: AgentState) -> bool:
    risk = state.get("risk", 0.0)
    return risk >= RISK_ESCALATE_THRESHOLD or (
        state.get("intent_label") == "complaint" and risk >= COMPLAINT_ESCALATE_THRESHOLD)


def escalation_trigger(flags: dict, state: AgentState) -> str:
    if flags.get("repeat_limit"):
        return "turn_limit"
    if flags.get("sop_gap"):
        return "sop_gap"
    keyword_hit = any(flags.get(k) for k in ("security_incident", "angry_keyword", "explicit_human", "repeated_failure"))
    model_hit = model_escalates(state)
    if keyword_hit and model_hit:
        return "both"
    return "keyword" if keyword_hit else "model"


def call_router(state: AgentState, flags: dict) -> tuple[str, str]:
    return route(
        state.get("intent_label", ""), state.get("emotion_label", "neutral"), state.get("risk", 0.0), flags,
        stage=state["stage"],
        qualification_complete=len(state.get("qualification", {})) >= len(QUALIFY_QUESTIONS),
    )


def decide(state: AgentState) -> tuple[AgentState, str, str, str]:
    """Run the router for the latest user message and apply the repeat guard.

    Returns (state, route, reason, trigger). trigger is "" unless escalating.
    """
    flags = keyword_flags(last_human_message(state) or "")
    action, reason = call_router(state, flags)

    last_decision = state.get("last_decision", "")
    repeat_count = state.get("decision_repeat_count", 0)
    repeat_count = repeat_count + 1 if action == last_decision and action not in ("escalate", "summary") else 1

    if repeat_count >= MAX_REPEAT_DECISIONS and action not in ("escalate", "summary"):
        flags["repeat_limit"] = f"'{action}' repeated {repeat_count} times without progress; forcing escalation."
        action, reason = call_router(state, flags)

    trigger = escalation_trigger(flags, state) if action == "escalate" else ""
    log_decision(state["session_id"], action, trigger, reason)
    return {**state, "last_decision": action, "decision_repeat_count": repeat_count}, action, reason, trigger


# ── Step 3: handlers ─────────────────────────────────────────────────────────
def escalation_response(state: AgentState, trigger: str) -> str:
    text = last_human_message(state) or ""
    if security_check(text)[0]:
        return ESCALATION_RESPONSES["security"]
    if sentiment_check(text)[0]:
        return ESCALATION_RESPONSES["upset"]
    if explicit_escalation(text)[0]:
        return ESCALATION_RESPONSES["human"]
    if repeated_failure_check(text)[0] or trigger == "turn_limit":
        return ESCALATION_RESPONSES["repeated"]
    if state.get("emotion_label") in ("angry", "frustrated", "sad") or state.get("intent_label") == "complaint":
        return ESCALATION_RESPONSES["upset"]
    return ESCALATION_RESPONSES["repeated"]


def escalate_handler(state: AgentState, reason: str, trigger: str, response: str | None = None) -> AgentState:
    """Hand off to a human: create a ticket, log, and reply."""
    response = response or escalation_response(state, trigger)
    qual = state.get("qualification", {})
    try:
        ticket = create_ticket(
            session_id=state["session_id"],
            severity=DEFAULT_ESCALATION_SEVERITY,
            issue_summary=qual.get("answer_2") or reason,
            employee=qual.get("answer_1"),
            device_system=qual.get("answer_3"),
        )
    except Exception:
        # Ticket creation must never break the escalation itself.
        ticket = None

    if ticket:
        response = f"{response} Ticket #{ticket['ticket_id']} created — Priority: {DEFAULT_ESCALATION_SEVERITY.title()}."

    result = assistant_payload({
        "response": response, "confidence": 1.0,
        "escalate": True, "escalation_reason": reason,
        "ticket_id": ticket["ticket_id"] if ticket else None,
        "ticket_severity": DEFAULT_ESCALATION_SEVERITY if ticket else None,
    })
    log_escalation(state["session_id"], reason, trigger, state.get("intent_label", ""), state.get("intent_confidence", 0.0))
    return {**state, "stage": "escalated", "escalation_reason": reason, "escalation_trigger": trigger,
            "messages": state["messages"] + [assistant_message(result)]}


def faq_handler(state: AgentState) -> AgentState:
    """Answer from the SOP. Groq writes the text; the router decides what
    happens if Groq reports the question is outside the SOP."""
    user_msg = last_human_message(state) or ""
    generated = llm_call(SYSTEM_BASE, format_messages(state))
    # Keep only the text-side fields; any routing-like keys Groq adds are ignored.
    result = assistant_payload({k: generated.get(k) for k in ("response", "confidence", "sop_gap")})

    if result["sop_gap"]:
        state = {**state, "unanswered_count": state["unanswered_count"] + 1,
                 "sop_gaps": state["sop_gaps"] + [user_msg]}
        action, reason = call_router(state, {"sop_gap": "Question outside SOP data"})
        log_decision(state["session_id"], action, "sop_gap", reason)
        return escalate_handler(state, reason, "sop_gap", response=result["response"])

    state = {**state, "unanswered_count": 0}
    starts_qualification = (state.get("intent_label") not in NON_SUPPORT_INTENTS
                            and user_msg.lower().strip() not in GREETINGS)
    if starts_qualification:
        result["response"] = f"{result['response']} {QUALIFY_QUESTIONS[0]}"
        state = {**state, "stage": "qualify", "qualification_started": True}

    return {**state, "messages": state["messages"] + [assistant_message(result)]}


def qualify_handler(state: AgentState) -> AgentState:
    """Collect the three structured qualification answers, one per turn."""
    user_msg = last_human_message(state) or ""
    new_qual = state["qualification"].copy()
    q_count = len(new_qual)

    def reply(text: str, stage_complete: bool = False) -> dict:
        return assistant_message(assistant_payload({"response": text, "confidence": 1.0, "stage_complete": stage_complete}))

    # Question 1 hasn't been asked yet, so this message is the employee's
    # original question rather than an answer — ask Question 1 instead.
    if not state.get("qualification_started") and q_count == 0:
        return {**state, "stage": "qualify", "qualification_started": True,
                "messages": state["messages"] + [reply(QUALIFY_QUESTIONS[0])]}

    # A reply that repeats the original question or an earlier answer
    # verbatim isn't answering the question just asked — re-ask instead of
    # letting qualification "complete" on copy-pasted non-answers.
    normalized = user_msg.strip().lower()
    already_said = {a.strip().lower() for a in new_qual.values()} | {
        m.get("content", "").strip().lower() for m in state["messages"][:-1] if m.get("role") == "user"
    }
    if q_count < len(QUALIFY_QUESTIONS) and normalized in already_said:
        text = f"That looks like something you already told me rather than an answer to this question: {QUALIFY_QUESTIONS[q_count]}"
        return {**state, "stage": "qualify", "messages": state["messages"] + [reply(text)]}

    if q_count < len(QUALIFY_QUESTIONS):
        new_qual[f"answer_{q_count + 1}"] = user_msg

    if len(new_qual) >= len(QUALIFY_QUESTIONS):
        message = reply("Thank you, that helps. I have your employee information, issue details, and device or "
                        "system details. I'll prepare a short session summary for Human IT Support.", True)
        next_stage = "summary"
    else:
        message = reply(QUALIFY_QUESTIONS[len(new_qual)])
        next_stage = "qualify"

    return {**state, "stage": next_stage, "qualification": new_qual,
            "messages": state["messages"] + [message]}


SUMMARY_PROMPT = """Based on this full conversation, generate a structured IT support session summary.

Qualification data collected: {qual}
SOP gaps identified: {gaps}

Also classify the severity of the business impact described in the qualification answers, using one of: "low", "medium", "high", "critical".
- low: minor inconvenience to a single person with an easy workaround. Examples: "my desktop wallpaper reset itself", "I'd like a second monitor".
- medium: a single person's work is slowed or annoying but they can still get work done. Examples: "printer on my floor won't connect and I have to walk to another one", "email is slow to load".
- high: blocks a person's core work, or affects multiple people. Examples: "I can't log into email at all and have a client deadline today", "the whole team's shared drive is inaccessible".
- critical: anything security-related — compromised credentials, suspected phishing/malware, unauthorized access, or a suspected breach. Examples: "I think I clicked a phishing link and entered my password", "someone logged into my account from a location I don't recognize".

Return ONLY valid JSON, no markdown fences:
{{
  "employee_intent": "<one sentence describing what the employee needs>",
  "key_details_collected": {{
    "employee": "<employee identifier or null>",
    "issue": "<issue and business impact or null>",
    "device_system": "<device, operating system, and error or null>"
  }},
  "sop_gaps": ["<question 1>", "<question 2>"],
  "recommended_next_action": "<what the team should do next>",
  "severity": "<one of: low, medium, high, critical>",
  "escalated": false,
  "escalation_reason": null
}}"""


def generate_summary(state: AgentState) -> dict:
    """Groq-written session summary, with a safe fallback built from what was collected."""
    qual = state.get("qualification", {})
    gaps = state.get("sop_gaps", [])
    system = SUMMARY_PROMPT.format(qual=json.dumps(qual), gaps=json.dumps(gaps))
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            reasoning_effort="low",
            messages=[{"role": "system", "content": system}] + format_messages(state)
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            raise ValueError("Empty response from Groq summary call")
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        try:
            return json.loads(raw)
        except Exception:
            return {"raw_summary": raw}
    except Exception as exc:
        print(f"[summary] Groq call failed, using fallback: {exc}")
        return {
            "employee_intent": qual.get("answer_2") or "Employee requested IT support; details unavailable due to a summary generation error.",
            "key_details_collected": {
                "employee": qual.get("answer_1"),
                "issue": qual.get("answer_2"),
                "device_system": qual.get("answer_3"),
            },
            "sop_gaps": gaps,
            "recommended_next_action": "Summary generation failed — a human IT support analyst should review the raw conversation directly.",
            "severity": "medium",
            "escalated": False,
            "escalation_reason": None,
            "degraded": True,
        }


def summary_handler(state: AgentState) -> AgentState:
    """Generate and save the structured session summary, and open a ticket."""
    qual = state.get("qualification", {})
    summary = generate_summary(state)

    save_final_summary(state["session_id"], {
        "intent_label": state["intent_label"],
        "intent_confidence": state["intent_confidence"],
        "emotion_label": state.get("emotion_label"),
        "risk": state.get("risk"),
        "escalation_trigger": state["escalation_trigger"],
        **summary,
    })

    severity = summary.get("severity") or "medium"
    key_details = summary.get("key_details_collected") or {}
    try:
        ticket = create_ticket(
            session_id=state["session_id"],
            severity=severity,
            issue_summary=key_details.get("issue") or qual.get("answer_2"),
            employee=key_details.get("employee") or qual.get("answer_1"),
            device_system=key_details.get("device_system") or qual.get("answer_3"),
        )
    except Exception:
        # Ticket creation must never break summary generation.
        ticket = None

    response_text = "Session summary saved for Human IT Support. Thank you for using the AI Support Assistant."
    if ticket:
        response_text = (
            f"Session summary saved for Human IT Support. Ticket #{ticket['ticket_id']} created "
            f"— Priority: {severity.title()}. Thank you for using the AI Support Assistant."
        )

    result = assistant_payload({
        "response": response_text,
        "confidence": 1.0,
        "stage_complete": True,
        "degraded": summary.get("degraded", False),
        "ticket_id": ticket["ticket_id"] if ticket else None,
        "ticket_severity": severity if ticket else None,
    })
    return {**state, "stage": "summary", "conversation_ended": True,
            "messages": state["messages"] + [assistant_message(result)]}


# ── Turn entry point ─────────────────────────────────────────────────────────
def run_turn(state: AgentState, msg: str) -> AgentState:
    """Process one employee message and return the new state.

    The returned state's messages are the old ones + the user message + zero
    or more assistant replies. No reply means the session has already been
    handed off (escalated or summarised).
    """
    state = {**state, "messages": state["messages"] + [user_message(msg)]}
    if state.get("conversation_ended") or state["stage"] == "escalated":
        return {**state, "conversation_ended": True}
    before = len(state["messages"])

    state = classify(state)
    state, action, reason, trigger = decide(state)

    if action == "escalate":
        state = escalate_handler(state, reason, trigger)
    elif action == "summary":
        state = summary_handler(state)
    elif action == "qualify":
        state = qualify_handler(state)
        # The third answer completes qualification — summarise in the same turn.
        if state["stage"] == "summary":
            state = summary_handler(state)
    else:
        state = faq_handler(state)
    return annotate_replies(state, before)


def classification_meta(state: AgentState) -> dict:
    return {
        "intent": state.get("intent_label", ""), "intent_confidence": state.get("intent_confidence", 0.0),
        "emotion": state.get("emotion_label", "neutral"), "emotion_confidence": state.get("emotion_confidence", 0.0),
        "risk": state.get("risk", 0.0), "classifier_degraded": state.get("intent_degraded", False),
    }


def annotate_replies(state: AgentState, start: int) -> AgentState:
    """Stamp this turn's classification onto its assistant replies, so it is
    persisted as message metadata and shown again when a session reloads."""
    meta = classification_meta(state)
    messages = state["messages"][:start] + [
        assistant_message({**json.loads(m["content"]), **meta}) if m.get("role") == "assistant" else m
        for m in state["messages"][start:]
    ]
    return {**state, "messages": messages}


def get_initial_state(session_id: str = None) -> AgentState:
    if not session_id:
        session_id = str(uuid.uuid4())
    return {
        "messages": [],
        "session_id": session_id,
        "stage": "faq",
        "unanswered_count": 0,
        "qualification": {},
        "qualification_started": False,
        "escalation_reason": "",
        "sop_gaps": [],
        "conversation_ended": False,
        "intent_label": "",
        "intent_confidence": 0.0,
        "emotion_label": "neutral",
        "emotion_confidence": 0.0,
        "risk": 0.0,
        "intent_source_message": "",
        "intent_degraded": False,
        "escalation_trigger": "",
        "last_decision": "",
        "decision_repeat_count": 0,
    }
