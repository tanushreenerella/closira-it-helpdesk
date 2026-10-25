"""
Closira AI IT Helpdesk Support Agent — plain-Python turn pipeline.

    run_turn(state, msg): classify -> route -> handler

Classification is the local ML model (backend/ml), routing is the
deterministic backend/router.py. Groq is used for three text jobs only:
the FAQ reply, slot extraction (pulling issue details out of a message) and
the end-of-session summary. Groq never chooses a route.

Support flow once an issue is reported:
    slot filling  -> ask only for missing slots, one at a time
    troubleshoot  -> offer the SOP self-help steps, ask if they worked
    ticket        -> only if still unresolved, or the employee asks for one

Messages are plain dicts: {"role": "user", "content": str} and
{"role": "assistant", "content": <JSON string of the reply payload>} — the
JSON-string convention is what database.py persists and restores.
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import TypedDict, Literal

from groq import Groq
from dotenv import load_dotenv
try:
    from .ml.inference import classify_message
    from .router import (route, risk_score, emotion_only_risk, keyword_flags, sentiment_check, explicit_escalation, repeated_failure_check,
                         security_check, ticket_severity, impact_level, NON_SUPPORT_INTENTS, RISK_ESCALATE_THRESHOLD,
                         COMPLAINT_ESCALATE_THRESHOLD, INTENT_CONFIDENCE_THRESHOLD, UNCERTAIN_INTENT, SLOT_ANSWER_INTENT)
    from .database import record_escalation, record_decision, save_final_summary, create_ticket
except ImportError:
    from ml.inference import classify_message
    from router import (route, risk_score, emotion_only_risk, keyword_flags, sentiment_check, explicit_escalation, repeated_failure_check,
                        security_check, ticket_severity, impact_level, NON_SUPPORT_INTENTS, RISK_ESCALATE_THRESHOLD,
                        COMPLAINT_ESCALATE_THRESHOLD, INTENT_CONFIDENCE_THRESHOLD, UNCERTAIN_INTENT, SLOT_ANSWER_INTENT)
    from database import record_escalation, record_decision, save_final_summary, create_ticket

load_dotenv()

# ── Load SOP ────────────────────────────────────────────────────────────────
with open(os.path.join(os.path.dirname(__file__), "sop.json"), "r", encoding="utf-8") as f:
    SOP = json.load(f)

SOP_TEXT = json.dumps(SOP, indent=2)

# ── Groq client ──────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"

# ── Config ───────────────────────────────────────────────────────────────────
TRIVIAL_MESSAGES = {"hi", "hello", "hey", "hiya", "ok", "okay", "thanks", "thank you", "yes", "no", "yep", "nope"}
GREETINGS = {"hi", "hello", "hey", "hiya"}
MAX_REPEAT_DECISIONS = 5  # loop guard: force escalation after this many identical non-terminal routes in a row

DECISION_LOG_PATH = os.path.join(os.path.dirname(__file__), "decision_log.json")

# ── Slots ────────────────────────────────────────────────────────────────────
SLOTS = ("requester", "issue_category", "problem_details", "error_message", "device_os", "impact")
# requester (name or employee ID) is asked first but optional: declining
# stores REQUESTER_SKIPPED so it is never asked again.
REQUESTER_SKIPPED = "not provided"
SKIP_ANSWER = re.compile(r"^\s*(skip|no|nope|no thanks|n/?a|pass|prefer not|i'?d rather not|rather not|"
                         r"anonymous|not now|none|no name)\b", re.I)

# Slot value -> SOP playbook "issue" name.
CATEGORIES = {
    "account": "Account and password",
    "network": "Wi-Fi and network",
    "vpn": "VPN",
    "device": "Device",
    "email": "Email",
    "software": "Software and access",
    "other": None,
}
PLAYBOOKS = {key: next((p for p in SOP["troubleshooting_playbooks"] if p["issue"] == name), None)
             for key, name in CATEGORIES.items()}

# Fallback category detection when extraction doesn't return one. Checked in order.
CATEGORY_KEYWORDS = [
    ("vpn", re.compile(r"\bvpn\b", re.I)),
    ("email", re.compile(r"\b(e-?mail|outlook|mailbox|inbox|webmail)\b", re.I)),
    ("account", re.compile(r"\b(password|log ?in|sign ?in|account|locked out|mfa|multi-factor|2fa)\b", re.I)),
    ("network", re.compile(r"\b(wi-?fi|network|internet|ethernet|wired|router|hotspot)\b", re.I)),
    ("software", re.compile(r"\b(software|install|licen[cs]e|application|app access|permission|access to)\b", re.I)),
    ("device", re.compile(r"\b(laptop|computer|pc|desktop|monitor|screen|keyboard|mouse|printer|battery|charger|device|phone)\b", re.I)),
]

SLOT_QUESTIONS = {
    "requester": {
        "default": "First, could I have your name or employee ID for the ticket? This is optional — say \"skip\" if you'd rather not.",
    },
    "issue_category": {
        "default": "Which area is this about: account or password, Wi-Fi or network, VPN, a device, email, or software and access?",
    },
    "problem_details": {
        "default": "Can you describe what is going wrong?",
        "vpn": "What happens when you try to connect to the VPN?",
        "network": "What happens when you try to get online?",
        "account": "What happens when you try to sign in?",
        "device": "What is the device doing (or not doing)?",
        "email": "What is going wrong with your email?",
        "software": "Which application or resource is it, and what happens when you try to use it?",
    },
    "error_message": {
        "default": "What exact error message do you see, if any? (Say \"none\" if there isn't one.)",
        "vpn": "What error does the VPN client show when it fails, if any? An error code helps.",
        "network": "Do you see any error, such as \"No internet\" or \"Can't connect to this network\"? (Say \"none\" if not.)",
        "account": "What message do you get on the sign-in screen, if any? Please don't share your password.",
        "device": "Is there any error message, warning light, or beep code? (Say \"none\" if not.)",
        "email": "Do you see an error in your mail client or when a message bounces? (Say \"none\" if not.)",
        "software": "What error or access-denied message do you see, if any?",
    },
    "device_os": {
        "default": "Which device and operating system are you using?",
        "vpn": "Which device and operating system are you connecting from?",
        "network": "Which device are you on, and is it on Wi-Fi, a wired connection, or both?",
        "account": "Which device and browser or app are you signing in from?",
        "device": "What kind of device is it (make or model if you know it), and which operating system?",
        "email": "Are you using the desktop mail app, webmail, or your phone — and on which operating system?",
        "software": "Which device and operating system do you need the software on?",
    },
    "impact": {
        "default": "How is this affecting your work — are you completely blocked, slowed down, or is it a minor inconvenience? Is anyone else affected?",
    },
}

# Asked once when problem_details is too vague to act on.
VAGUE_FOLLOWUPS = {
    "default": "Could you be more specific about what happens — what you expect to see and what you see instead?",
    "vpn": "Could you be more specific: does the VPN fail to connect at all, connect and then drop, or connect but you can't reach internal sites?",
    "network": "Could you be more specific: no connection at all, a slow connection, or one that keeps dropping? Is it Wi-Fi, wired, or both?",
    "account": "Could you be more specific: are you unable to sign in, locked out, or stuck at the multi-factor step?",
    "device": "Could you be more specific: does it not power on, crash or freeze, run very slowly, or is a part (screen, keyboard, battery) failing?",
    "email": "Could you be more specific: is the problem sending, receiving, signing in, or opening a shared mailbox?",
    "software": "Could you be more specific: does it fail to install, fail to open, or say you don't have access?",
}

VAGUE_DETAILS = re.compile(
    r"\b(not working( properly| right| correctly)?|doesn'?t work|isn'?t working|won'?t work|stopped working|broken|"
    r"acting (up|weird|strange)|weird|messed up|something('?s| is) wrong|not (right|good)|glitch\w*|bugg?y|"
    r"(an? )?(issue|problem|trouble)s?( with)?|help)\b", re.I)
SPECIFIC_HINT = re.compile(r"\d|\b(error|code|when i|after|every time|says|shows|message|drops?|crash\w*|freez\w*|"
                           r"slow|locked|denied|bounce\w*|won'?t (connect|open|start|turn on|load|send|sync))\b", re.I)

FIX_UNRESOLVED = re.compile(
    r"\b(no|nope|nah|didn'?t|did not|doesn'?t|does not|still|not (working|fixed|resolved)|same (issue|problem|thing)|"
    r"nothing|no luck|not really)\b", re.I)
FIX_RESOLVED = re.compile(r"\b(yes|yep|yeah|yup|worked|works( now)?|fixed|resolved|sorted|all good|that did it|perfect)\b", re.I)


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
    stage: Literal["faq", "qualify", "troubleshoot", "escalated", "summary"]
    unanswered_count: int
    qualification: dict          # filled slots, keyed by SLOTS
    qualification_started: bool
    pending_slot: str            # slot the last question asked for ("" if none)
    details_vague: bool          # problem_details needs a follow-up
    followup_asked: bool         # the vague-details follow-up was already asked
    impact_level: str            # blocked | degraded | minor | ""
    troubleshoot_reasked: bool
    ready_for_ticket: bool
    escalation_reason: str
    sop_gaps: list
    conversation_ended: bool
    intent_label: str            # model intent, "uncertain", or "slot_answer"
    intent_raw: str              # model intent before the uncertain / slot-answer relabel
    intent_confidence: float
    emotion_label: str
    emotion_confidence: float
    risk: float
    peak_risk: float             # highest per-turn risk this session; drives ticket severity
    intent_source_message: str
    intent_degraded: bool
    escalation_trigger: str
    last_decision: str
    decision_repeat_count: int
    last_progress: str


# ── Prompts ──────────────────────────────────────────────────────────────────
SYSTEM_BASE = f"""You are the AI Support Assistant for Closira IT Helpdesk, providing employee IT support.

YOUR STRICT RULES:
1. Answer ONLY from the SOP data below. Do NOT invent systems, access, policies, troubleshooting results, or any facts.
2. If a question is outside the SOP, say clearly: "I don't have that information right now", offer to connect them with Human IT Support, and set sop_gap=true.
3. Never ask for passwords, one-time codes, recovery codes, or sensitive email content.
4. Be concise, practical, and professional — like a knowledgeable IT support analyst.
5. Do not ask the employee follow-up questions; the assistant asks for any missing details separately.
6. Always respond in English.

SOP DATA:
{SOP_TEXT}

RESPONSE FORMAT (always return valid JSON, nothing else, no markdown fences):
{{
  "response": "<your reply to the employee>",
  "sop_gap": <true/false>
}}
"""

EXTRACT_PROMPT = """You extract structured IT-support details from the employee's LATEST message. You do not reply to the employee and you do not decide what happens next.

Question the employee is answering (may be empty): {question}
Details already collected: {known}

Return ONLY valid JSON, no markdown fences:
{{
  "requester": "<the employee's name and/or employee ID>", or "not provided" if they decline to give it, or null,
  "issue_category": one of {categories} or null,
  "problem_details": "<what is going wrong, in the employee's words>" or null,
  "error_message": "<exact error text or code>", or "none" if they say there is no error, or null,
  "device_os": "<device and operating system>" or null,
  "impact": "<how it affects their work, and who else is affected>" or null,
  "impact_level": "blocked" | "degraded" | "minor" | null,
  "problem_details_vague": <true if problem_details only says something is broken / not working without saying what actually happens, else false>
}}
Only fill a field if the latest message states it. Never guess or invent values. Never include passwords or codes."""

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


def groq_json(system: str, messages: list, max_tokens: int = 512) -> dict:
    """Call Groq and parse a JSON object from its reply. Raises on failure."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}] + messages
    )
    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def llm_call(system: str, messages: list) -> dict:
    """Groq FAQ reply. Only response and sop_gap are read from the result."""
    try:
        parsed = groq_json(system, messages)
    except json.JSONDecodeError as exc:
        return {"response": exc.doc.strip(), "sop_gap": False}
    # Clean up response field if Groq appended a JSON blob inside it
    resp = str(parsed.get("response", ""))
    if "{" in resp:
        parsed["response"] = resp[:resp.index("{")].strip()
    return parsed


def llm_extract(system: str, messages: list) -> dict:
    """Groq slot extraction. Returns the parsed JSON; raises on failure."""
    return groq_json(system, messages, max_tokens=300)


def assistant_payload(result: dict) -> dict:
    """Keep assistant messages structured so the UI can read routing data."""
    return {
        "response": str(result.get("response", "")).strip(),
        "escalate": bool(result.get("escalate", False)),
        "escalation_reason": result.get("escalation_reason"),
        "stage_complete": bool(result.get("stage_complete", False)),
        "sop_gap": bool(result.get("sop_gap", False)),
        "degraded": bool(result.get("degraded", False)),
        "pending_slot": result.get("pending_slot"),
        "ticket_id": result.get("ticket_id"),
        "ticket_severity": result.get("ticket_severity"),
        "resolved": bool(result.get("resolved", False)),
    }


def reply(state: AgentState, text: str, **fields) -> AgentState:
    return {**state, "messages": state["messages"] + [assistant_message(assistant_payload({"response": text, **fields}))]}


def last_human_message(state: AgentState) -> str | None:
    """Content of the most recent user message, regardless of any assistant
    replies appended after it this turn."""
    for m in reversed(state["messages"]):
        if m.get("role") == "user":
            return m.get("content", "")
    return None


def last_assistant_text(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if m.get("role") == "assistant":
            try:
                return json.loads(m["content"]).get("response", "")
            except Exception:
                return m.get("content", "")
    return ""


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
def neutral_prediction(degraded: bool = False, intent: str = "information_request", confidence: float = 0.0) -> dict:
    return {"intent": intent, "intent_confidence": confidence, "intent_probs": None,
            "emotion": "neutral", "emotion_confidence": 0.0, "emotion_probs": None,
            "degraded": degraded}


def classify_text(text: str) -> dict:
    if text.lower().strip() in TRIVIAL_MESSAGES:
        return neutral_prediction(intent="chit_chat", confidence=1.0)
    try:
        return classify_message(text)
    except Exception as exc:
        print(f"[classifier] Failed to classify: {exc}")
        return neutral_prediction(degraded=True)


def answering_question(state: AgentState) -> bool:
    """The latest message answers a slot or troubleshooting question."""
    return state["stage"] == "troubleshoot" or (state["stage"] == "qualify" and bool(state.get("pending_slot")))


def classify(state: AgentState) -> AgentState:
    """Attach intent / emotion / risk for the latest user message to state.

    - Slot / troubleshooting answers skip intent: it is labelled
      "slot_answer".
    - An intent below INTENT_CONFIDENCE_THRESHOLD is labelled "uncertain".
    Both are left out of routing, and their risk comes from emotion + emotion
    keyword rules only (router.emotion_only_risk), so a confident frustrated
    or angry message can still escalate on its own.
    Re-uses the cached prediction only when it was computed for THIS message
    (tracked via intent_source_message).
    """
    text = last_human_message(state)
    if text is None or state.get("intent_source_message") == text:
        return state
    p = classify_text(text)
    intent, confidence, intent_probs = p["intent"], p["intent_confidence"], p.get("intent_probs")
    if answering_question(state):
        intent, intent_probs = SLOT_ANSWER_INTENT, None
    elif confidence < INTENT_CONFIDENCE_THRESHOLD:
        intent, intent_probs = UNCERTAIN_INTENT, None
    if intent in (SLOT_ANSWER_INTENT, UNCERTAIN_INTENT):
        risk = emotion_only_risk(p["emotion"], p.get("emotion_probs"), p["emotion_confidence"], keyword_flags(text))
    else:
        risk = risk_score(intent, p["emotion"], intent_probs, p.get("emotion_probs"))
    return {
        **state,
        "intent_label": intent, "intent_raw": p["intent"], "intent_confidence": confidence,
        "emotion_label": p["emotion"], "emotion_confidence": p["emotion_confidence"],
        "risk": risk, "peak_risk": max(state.get("peak_risk", 0.0), risk),
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
        stage=state["stage"], qualification_complete=state.get("ready_for_ticket", False),
    )


def progress_signature(state: AgentState) -> str:
    filled = sum(1 for s in SLOTS if state.get("qualification", {}).get(s))
    return f"{state['stage']}:{filled}:{int(state.get('followup_asked', False))}"


def decide(state: AgentState) -> tuple[AgentState, str, str, str]:
    """Run the router for the latest user message and apply the repeat guard.

    The guard counts consecutive identical routes, but only forces escalation
    when the previous turn made no progress (no new slot, no stage change) —
    the slot-filling flow legitimately routes "qualify" several turns in a row.
    Returns (state, route, reason, trigger). trigger is "" unless escalating.
    """
    flags = keyword_flags(last_human_message(state) or "")
    action, reason = call_router(state, flags)

    progress = progress_signature(state)
    made_progress = progress != state.get("last_progress")
    last_decision = state.get("last_decision", "")
    repeat_count = state.get("decision_repeat_count", 0)
    repeat_count = repeat_count + 1 if action == last_decision and action not in ("escalate", "summary") else 1

    if repeat_count >= MAX_REPEAT_DECISIONS and action not in ("escalate", "summary") and not made_progress:
        flags["repeat_limit"] = f"'{action}' repeated {repeat_count} times without progress; forcing escalation."
        action, reason = call_router(state, flags)

    trigger = escalation_trigger(flags, state) if action == "escalate" else ""
    log_decision(state["session_id"], action, trigger, reason)
    return ({**state, "last_decision": action, "decision_repeat_count": repeat_count, "last_progress": progress},
            action, reason, trigger)


# ── Slot filling ─────────────────────────────────────────────────────────────
def guess_category(text: str) -> str | None:
    return next((cat for cat, rx in CATEGORY_KEYWORDS if rx.search(text)), None)


def is_vague(details: str) -> bool:
    """Short "it's not working"-style descriptions that say nothing about what happens."""
    words = len(re.findall(r"\w+", details))
    return words <= 10 and bool(VAGUE_DETAILS.search(details)) and not SPECIFIC_HINT.search(details)


def _clean(value) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return None if value.lower() in ("", "null", "none given", "unknown", "n/a") else value


def extract_slots(state: AgentState, text: str) -> dict:
    """Slot values stated in `text` (Groq extraction + deterministic fallbacks).

    Returns {"slots": {...}, "vague": bool, "impact_level": str | None}.
    """
    pending = state.get("pending_slot") or ""
    known = {k: v for k, v in state.get("qualification", {}).items() if v}
    system = EXTRACT_PROMPT.format(question=last_assistant_text(state) if pending else "",
                                   known=json.dumps(known), categories=json.dumps(list(CATEGORIES)))
    try:
        raw = llm_extract(system, [{"role": "user", "content": text}])
        if not isinstance(raw, dict):
            raise ValueError("extraction did not return an object")
    except Exception as exc:
        print(f"[extract] Groq extraction failed, using fallback: {exc}")
        raw = {}

    slots = {s: _clean(raw.get(s)) for s in SLOTS}
    if slots["issue_category"] not in CATEGORIES:
        slots["issue_category"] = None
    slots["issue_category"] = slots["issue_category"] or guess_category(text)
    if pending == "requester" and (SKIP_ANSWER.search(text) or (slots["requester"] or "").lower() == REQUESTER_SKIPPED):
        slots["requester"] = REQUESTER_SKIPPED
    # Whatever the employee said answers the question they were asked, even
    # if extraction couldn't structure it — never re-ask the same slot forever.
    if pending and not slots.get(pending):
        slots[pending] = (guess_category(text) or "other") if pending == "issue_category" else text.strip()
    # The first message of an issue is its description.
    if not pending and not known.get("problem_details") and not slots["problem_details"]:
        slots["problem_details"] = text.strip()
    level = raw.get("impact_level") if raw.get("impact_level") in ("blocked", "degraded", "minor") else None
    return {"slots": slots, "vague": bool(raw.get("problem_details_vague")), "impact_level": level}


def collect_slots(state: AgentState, text: str) -> AgentState:
    """Merge the slots stated in `text` into state. Filled slots are never
    overwritten, except that a vague-details follow-up adds to problem_details."""
    extracted = extract_slots(state, text)
    qual = dict(state.get("qualification", {}))
    pending = state.get("pending_slot") or ""
    details_vague = state.get("details_vague", False)

    followup_answer = pending == "problem_details" and state.get("followup_asked") and qual.get("problem_details")
    if followup_answer:
        qual["problem_details"] = f"{qual['problem_details']} — {extracted['slots']['problem_details'] or text.strip()}"
        details_vague = False

    for slot in SLOTS:
        value = extracted["slots"].get(slot)
        if value and not qual.get(slot):
            qual[slot] = value
            if slot == "problem_details":
                details_vague = extracted["vague"] or is_vague(value)

    level = state.get("impact_level") or extracted["impact_level"] or impact_level(qual.get("impact")) or ""
    return {**state, "qualification": qual, "details_vague": details_vague, "impact_level": level}


def next_question(state: AgentState) -> tuple[str | None, str | None]:
    """(slot, question) for the next thing to ask, or (None, None) when done."""
    qual = state.get("qualification", {})
    category = qual.get("issue_category") or "default"
    if not qual.get("requester"):
        return "requester", SLOT_QUESTIONS["requester"]["default"]
    if qual.get("problem_details") and state.get("details_vague") and not state.get("followup_asked"):
        return "problem_details", VAGUE_FOLLOWUPS.get(category, VAGUE_FOLLOWUPS["default"])
    for slot in SLOTS:
        if not qual.get(slot):
            questions = SLOT_QUESTIONS[slot]
            return slot, questions.get(category, questions["default"])
    return None, None


def self_help_steps(category: str | None) -> list[str]:
    playbook = PLAYBOOKS.get(category or "")
    return list(playbook.get("self_help_steps", [])) if playbook else []


def ask_next(state: AgentState, prefix: str = "") -> AgentState:
    """Ask for the next missing slot, or — once all are filled — offer the
    SOP self-help steps before any ticket is created."""
    lead = f"{prefix} " if prefix else ""
    slot, question = next_question(state)
    if slot:
        followup = slot == "problem_details" and bool(state.get("qualification", {}).get("problem_details"))
        state = {**state, "stage": "qualify", "qualification_started": True, "pending_slot": slot,
                 "followup_asked": state.get("followup_asked", False) or followup}
        return reply(state, f"{lead}{question}", pending_slot=slot)

    steps = self_help_steps(state["qualification"].get("issue_category"))
    if not steps:
        state = {**state, "pending_slot": "", "ready_for_ticket": True}
        return reply(state, f"{lead}Thanks — I have everything I need. Our SOP has no self-service fix for this, so I'll raise a ticket for Human IT Support.",
                     stage_complete=True)

    numbered = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))
    state = {**state, "stage": "troubleshoot", "pending_slot": ""}
    return reply(state, f"{lead}Thanks, that's everything I need. Before I raise a ticket, please try these steps from our IT SOP:\n"
                        f"{numbered}\nDid that fix the problem? (yes / no)")


def fix_outcome(text: str) -> str | None:
    """resolved | unresolved | None (unclear) for a reply to "did that fix it?"."""
    if FIX_UNRESOLVED.search(text):
        return "unresolved"
    if FIX_RESOLVED.search(text):
        return "resolved"
    return None


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


def severity_for(state: AgentState) -> str:
    """Ticket priority from the session's peak risk + the reported impact."""
    return ticket_severity(state.get("peak_risk", state.get("risk", 0.0)),
                           state.get("qualification", {}).get("impact"),
                           level=state.get("impact_level") or None,
                           security=security_check(last_human_message(state) or "")[0])


def ticket_fields(state: AgentState, fallback_issue: str | None = None) -> dict:
    q = state.get("qualification", {})
    category = CATEGORIES.get(q.get("issue_category") or "") or q.get("issue_category")
    issue = " — ".join(x for x in (category, q.get("problem_details"),
                                   q.get("impact") and f"Impact: {q['impact']}") if x)
    device = "; ".join(x for x in (q.get("device_os"), q.get("error_message") and f"Error: {q['error_message']}") if x)
    requester = q.get("requester")
    return {"issue_summary": issue or fallback_issue,
            "employee": None if requester in (None, "", REQUESTER_SKIPPED) else requester,
            "device_system": device or None}


def open_ticket(state: AgentState, severity: str, fallback_issue: str | None = None) -> dict | None:
    try:
        return create_ticket(session_id=state["session_id"], severity=severity, **ticket_fields(state, fallback_issue))
    except Exception:
        # Ticket creation must never break the turn.
        return None


def escalate_handler(state: AgentState, reason: str, trigger: str, response: str | None = None) -> AgentState:
    """Hand off to a human: create a ticket, log, and reply."""
    response = response or escalation_response(state, trigger)
    severity = severity_for(state)
    ticket = open_ticket(state, severity, fallback_issue=reason)
    if ticket:
        response = f"{response} Ticket #{ticket['ticket_id']} created — Priority: {severity.title()}."

    log_escalation(state["session_id"], reason, trigger, state.get("intent_label", ""), state.get("intent_confidence", 0.0))
    state = {**state, "stage": "escalated", "escalation_reason": reason, "escalation_trigger": trigger, "pending_slot": ""}
    return reply(state, response, escalate=True, escalation_reason=reason,
                 ticket_id=ticket["ticket_id"] if ticket else None, ticket_severity=severity if ticket else None)


def faq_handler(state: AgentState) -> AgentState:
    """Answer from the SOP. Groq writes the text; the router decides what
    happens if Groq reports the question is outside the SOP. A support
    request then moves into slot filling."""
    user_msg = last_human_message(state) or ""
    generated = llm_call(SYSTEM_BASE, format_messages(state))
    # Keep only the text-side fields; any routing-like keys Groq adds are ignored.
    answer, sop_gap = str(generated.get("response", "")).strip(), bool(generated.get("sop_gap"))

    if sop_gap:
        state = {**state, "unanswered_count": state["unanswered_count"] + 1,
                 "sop_gaps": state["sop_gaps"] + [user_msg]}
        action, reason = call_router(state, {"sop_gap": "Question outside SOP data"})
        log_decision(state["session_id"], action, "sop_gap", reason)
        return escalate_handler(state, reason, "sop_gap", response=answer)

    state = {**state, "unanswered_count": 0}
    starts_qualification = (state.get("intent_label") not in NON_SUPPORT_INTENTS
                            and user_msg.lower().strip() not in GREETINGS)
    if not starts_qualification:
        return reply(state, answer)
    return ask_next(collect_slots(state, user_msg), prefix=answer)


def qualify_handler(state: AgentState) -> AgentState:
    """Slot filling, then the "did the SOP fix work?" check."""
    user_msg = last_human_message(state) or ""

    if state["stage"] == "troubleshoot":
        outcome = fix_outcome(user_msg)
        if outcome == "resolved":
            return {**state, "ready_for_ticket": False, "resolved": True}
        if outcome == "unresolved" or state.get("troubleshoot_reasked"):
            state = {**state, "ready_for_ticket": True}
            return reply(state, "Thanks for trying those steps. I'll raise a ticket so Human IT Support can take it from here.",
                         stage_complete=True)
        state = {**state, "troubleshoot_reasked": True}
        return reply(state, "Did those steps fix the problem? Please answer yes or no.")

    # Autonomous routing can land here before anything was asked; treat the
    # message as the issue description, like the FAQ hand-off does.
    if not state.get("pending_slot") and not state.get("qualification"):
        return ask_next(collect_slots(state, user_msg))

    # A reply that repeats an earlier message verbatim isn't answering the
    # question just asked — re-ask instead of filling the slot with it.
    normalized = user_msg.strip().lower()
    earlier = {m.get("content", "").strip().lower() for m in state["messages"][:-1] if m.get("role") == "user"}
    _, question = next_question(state)
    if question and normalized in earlier:
        return reply(state, f"That looks like something you already told me rather than an answer to this question: {question}",
                     pending_slot=state.get("pending_slot"))

    return ask_next(collect_slots(state, user_msg))


SUMMARY_PROMPT = """Based on this full conversation, generate a structured IT support session summary.

Issue details collected: {qual}
SOP gaps identified: {gaps}
Outcome: {outcome}

Return ONLY valid JSON, no markdown fences:
{{
  "employee_intent": "<one sentence describing what the employee needs>",
  "key_details_collected": {{
    "requester": "<name or employee ID, or null>",
    "issue_category": "<category or null>",
    "problem_details": "<what is going wrong or null>",
    "error_message": "<error or null>",
    "device_os": "<device and operating system or null>",
    "impact": "<business impact or null>"
  }},
  "sop_gaps": ["<question 1>", "<question 2>"],
  "recommended_next_action": "<what the team should do next>",
  "escalated": false,
  "escalation_reason": null
}}"""


def generate_summary(state: AgentState, outcome: str = "unresolved — ticket raised") -> dict:
    """Groq-written session summary, with a safe fallback built from what was collected."""
    qual = state.get("qualification", {})
    gaps = state.get("sop_gaps", [])
    system = SUMMARY_PROMPT.format(qual=json.dumps(qual), gaps=json.dumps(gaps), outcome=outcome)
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
            "employee_intent": qual.get("problem_details") or "Employee requested IT support; details unavailable due to a summary generation error.",
            "key_details_collected": {slot: qual.get(slot) for slot in SLOTS},
            "sop_gaps": gaps,
            "recommended_next_action": "Summary generation failed — a human IT support analyst should review the raw conversation directly.",
            "escalated": False,
            "escalation_reason": None,
            "degraded": True,
        }


def summary_handler(state: AgentState, resolved: bool = False) -> AgentState:
    """Save the session summary. Opens a ticket unless the SOP fix resolved it."""
    summary = generate_summary(state, "resolved by SOP self-help steps" if resolved else "unresolved — ticket raised")
    severity = None if resolved else severity_for(state)

    save_final_summary(state["session_id"], {
        "intent_label": state["intent_label"],
        "intent_confidence": state["intent_confidence"],
        "emotion_label": state.get("emotion_label"),
        "peak_risk": state.get("peak_risk"),
        "escalation_trigger": state["escalation_trigger"],
        "resolved": resolved,
        "severity": severity,
        **summary,
    })

    state = {**state, "stage": "summary", "conversation_ended": True, "pending_slot": ""}
    if resolved:
        return reply(state, "Great, I'm glad that fixed it. I've saved a short session summary — no ticket was needed. "
                            "Thank you for using the AI Support Assistant.",
                     stage_complete=True, resolved=True, degraded=summary.get("degraded", False))

    ticket = open_ticket(state, severity)
    text = "Session summary saved for Human IT Support. Thank you for using the AI Support Assistant."
    if ticket:
        text = (f"Session summary saved for Human IT Support. Ticket #{ticket['ticket_id']} created "
                f"— Priority: {severity.title()}. Thank you for using the AI Support Assistant.")
    return reply(state, text, stage_complete=True, degraded=summary.get("degraded", False),
                 ticket_id=ticket["ticket_id"] if ticket else None, ticket_severity=severity if ticket else None)


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
    else:
        state = qualify_handler(state) if action == "qualify" else faq_handler(state)
        # The flow can finish within this turn: the SOP fix worked, or
        # everything is collected and a ticket is due.
        if state.pop("resolved", False):
            state = summary_handler(state, resolved=True)
        elif state.get("ready_for_ticket") and not state.get("conversation_ended"):
            state = summary_handler(state)
    return annotate_replies(state, before)


def classification_meta(state: AgentState) -> dict:
    return {
        "intent": state.get("intent_label", ""), "intent_raw": state.get("intent_raw", ""),
        "intent_confidence": state.get("intent_confidence", 0.0),
        "emotion": state.get("emotion_label", "neutral"), "emotion_confidence": state.get("emotion_confidence", 0.0),
        "risk": state.get("risk", 0.0), "peak_risk": state.get("peak_risk", 0.0),
        "classifier_degraded": state.get("intent_degraded", False),
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


def restore_flow_state(state: AgentState) -> AgentState:
    """Re-derive slot-filling bookkeeping that isn't stored in its own DB
    column (used when a session is reloaded from Postgres)."""
    pending = ""
    for m in reversed(state["messages"]):
        if m.get("role") == "assistant":
            try:
                pending = json.loads(m["content"]).get("pending_slot") or ""
            except Exception:
                pass
            break
    risks = []
    for m in state["messages"]:
        if m.get("role") == "assistant":
            try:
                risks.append(float(json.loads(m["content"]).get("risk") or 0.0))
            except Exception:
                pass
    restored = {**state, "pending_slot": pending if state["stage"] == "qualify" else "",
                "peak_risk": max(risks + [state.get("risk", 0.0)]),
                "impact_level": impact_level(state.get("qualification", {}).get("impact")) or ""}
    return {**restored, "last_progress": progress_signature(restored)}


def get_initial_state(session_id: str = None) -> AgentState:
    if not session_id:
        session_id = str(uuid.uuid4())
    state = {
        "messages": [],
        "session_id": session_id,
        "stage": "faq",
        "unanswered_count": 0,
        "qualification": {},
        "qualification_started": False,
        "pending_slot": "",
        "details_vague": False,
        "followup_asked": False,
        "impact_level": "",
        "troubleshoot_reasked": False,
        "ready_for_ticket": False,
        "escalation_reason": "",
        "sop_gaps": [],
        "conversation_ended": False,
        "intent_label": "",
        "intent_raw": "",
        "intent_confidence": 0.0,
        "emotion_label": "neutral",
        "emotion_confidence": 0.0,
        "risk": 0.0,
        "peak_risk": 0.0,
        "intent_source_message": "",
        "intent_degraded": False,
        "escalation_trigger": "",
        "last_decision": "",
        "decision_repeat_count": 0,
    }
    return {**state, "last_progress": progress_signature(state)}
