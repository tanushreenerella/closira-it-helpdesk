"""
Closira AI IT Helpdesk Support Agent
Built with LangGraph + Groq (LLaMA 3.3 70B)
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from groq import Groq
from dotenv import load_dotenv
try:
    from .nlp.classifier import classify_employee_message
    from .database import record_escalation, record_decision, save_final_summary, create_ticket
except ImportError:
    from nlp.classifier import classify_employee_message
    from database import record_escalation, record_decision, save_final_summary, create_ticket

load_dotenv()

# ── Load SOP ────────────────────────────────────────────────────────────────
with open(os.path.join(os.path.dirname(__file__), "sop.json"), "r") as f:
    SOP = json.load(f)

SOP_TEXT = json.dumps(SOP, indent=2)

# ── Groq client ──────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"

# ── Escalation log ───────────────────────────────────────────────────────────
ESCALATION_CONFIDENCE_THRESHOLD = 0.8
TRIVIAL_MESSAGES = {"hi", "hello", "hey", "hiya", "ok", "okay", "thanks", "thank you", "yes", "no", "yep", "nope"}
# escalation_result() escalates before summary_node's Groq call ever classifies
# severity, so there's nothing "already classified" to use yet — default to
# the same neutral middle value the summary fallback uses.
DEFAULT_ESCALATION_SEVERITY = "medium"

def log_escalation(session_id: str, reason: str, conversation: list, trigger: str = "groq",
                   intent_label: str = "normal", intent_confidence: float = 0.0):
    record_escalation(session_id, reason, trigger, intent_label, intent_confidence)

# ── NEW: Autonomous routing config ───────────────────────────────────────────
AUTONOMOUS_ROUTING = os.environ.get("AUTONOMOUS_ROUTING", "true").strip().lower() == "true"
DECISION_ACTIONS = ["faq", "qualify", "escalate", "summary"]
MAX_REPEAT_DECISIONS = 5  # loop guard: force escalation after this many identical non-terminal decisions in a row

DECISION_LOG_PATH = os.path.join(os.path.dirname(__file__), "decision_log.json")

def log_decision(session_id: str, chosen: str, overridden_by: str, reason: str):
    """Persist a decision node event to Postgres (routing_decisions table).

    Also appends to the local decision_log.json file for backward
    compatibility during the transition — that file doesn't survive a
    deployed host with an ephemeral filesystem, so Postgres is now the
    durable copy. Neither write may ever break the graph.
    """
    try:
        record_decision(session_id, chosen, overridden_by, reason)
    except Exception:
        pass

    entry = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chosen_action": chosen,
        "overridden_by": overridden_by,  # "" if not overridden, else "distilbert"/"keyword"/"validation_fallback"/"turn_limit"
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
        # Logging must never break the graph
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
    intent_source_message: str
    intent_degraded: bool
    escalation_trigger: str
    # NEW: loop guard for autonomous decision node
    last_decision: str
    decision_repeat_count: int

# ── System prompts ────────────────────────────────────────────────────────────
SYSTEM_BASE = f"""You are the AI Support Assistant for Closira IT Helpdesk, providing employee IT support.

YOUR STRICT RULES:
1. Answer ONLY from the SOP data below. Do NOT invent systems, access, policies, troubleshooting results, or any facts.
2. If a question is outside the SOP, say clearly: "I don't have that information right now" and offer to escalate.
3. Never ask for passwords, one-time codes, recovery codes, or sensitive email content. Suspected phishing, malware, or account compromise must be escalated immediately.
4. Be concise, practical, and professional — like a knowledgeable IT support analyst.
5. Always respond in English.

SOP DATA:
{SOP_TEXT}

RESPONSE FORMAT (always return valid JSON, nothing else, no markdown fences):
{{
  "response": "<your reply to the employee>",
  "confidence": <0.0-1.0 float>,
  "escalate": <true/false>,
  "escalation_reason": "<reason if escalate=true, else null>",
  "stage_complete": <true/false>,
  "sop_gap": <true/false>
}}
"""

QUALIFY_QUESTIONS = [
    "What is your name or employee identifier, if you are comfortable sharing it?",
    "Please describe the issue and the impact it is having on your work.",
    "What device and operating system are you using, and what exact error message do you see?"
]

QUALIFY_PROMPT = """You are now in IT issue qualification mode. Your goal is to ask the employee 3 structured questions ONE AT A TIME to collect the information required for support. The 3 questions are:
1. What is your name or employee identifier, if you are comfortable sharing it?
2. Please describe the issue and the impact it is having on your work.
3. What device and operating system are you using, and what exact error message do you see?

Ask only one question at a time. Never ask for a password, one-time code, or other secret. Once you have collected all 3 answers, set stage_complete=true and summarise what was collected in your response.

Still follow the same JSON response format and escalation rules."""

# NEW: decision node prompt
DECISION_SYSTEM_PROMPT = """You are a routing controller for an IT helpdesk agent. Based on the conversation and the state summary given, choose the SINGLE most appropriate next action.

Valid actions (choose exactly one):
- "faq": answer or troubleshoot the employee's question using SOP data
- "qualify": collect structured issue details (name, issue description, device/OS)
- "escalate": hand off to a human IT support analyst
- "summary": wrap up and produce a session summary (only appropriate once qualification is complete)

If qualification is already complete (all 3 details have been collected — check the qualification progress in the state summary), you MUST choose "summary". Do NOT choose "faq" at this point, even if you believe you could now answer the original question — the workflow requires generating a structured handoff summary once qualification is complete, not returning to troubleshooting.

Return ONLY valid JSON, no markdown fences:
{
  "action": "<one of: faq, qualify, escalate, summary>",
  "reasoning": "<one short sentence>"
}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def llm_call(system: str, messages: list) -> dict:
    """Call Groq and parse structured JSON response."""
    groq_messages = [{"role": "system", "content": system}] + messages
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=groq_messages
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        # Clean up response field if Groq appended JSON blob inside it
        resp = str(parsed.get("response", ""))
        if "{" in resp:
            parsed["response"] = resp[:resp.index("{")].strip()
        return parsed
    except json.JSONDecodeError:
        return {
            "response": raw,
            "confidence": 0.5,
            "escalate": False,
            "escalation_reason": None,
            "stage_complete": False,
            "sop_gap": False
        }

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

def sentiment_check(text: str) -> tuple[bool, str]:
    """Quick sentiment guard — returns (is_negative, reason)."""
    anger_keywords = [
        "angry", "furious", "terrible", "horrible", "disgusting", "lawsuit",
        "unacceptable", "scam", "fraud", "worst", "never coming back",
        "this is ridiculous", "i demand", "incompetent", "hate"
    ]
    text_lower = text.lower()
    for kw in anger_keywords:
        if kw in text_lower:
            return True, f"Angry sentiment detected: '{kw}'"
    return False, ""

def explicit_escalation(text: str) -> tuple[bool, str]:
    """Check if user explicitly asks for a human."""
    phrases = ["speak to a human", "talk to someone", "real person",
               "human", "human agent", "person", "representative", "support agent",
               "manager", "supervisor", "speak to staff", "transfer me to it support"]
    text_lower = text.lower()
    for p in phrases:
        if p in text_lower:
            return True, "Customer requested human agent"
    return False, ""

def repeated_failure_check(text: str) -> tuple[bool, str]:
    """Check for repeated troubleshooting attempts that have not resolved the issue."""
    failure_keywords = [
        "already tried", "tried again", "tried multiple times",
        "still doesn't work", "still confused", "still do not understand", "nothing is working",
    ]
    text_lower = text.lower()
    for keyword in failure_keywords:
        if keyword in text_lower:
            return True, f"Repeated failure signal detected: '{keyword}'"
    return False, ""

def last_human_message(state: AgentState) -> str | None:
    """Content of the most recent HUMAN message, regardless of what an
    assistant node may have appended after it this turn."""
    for m in reversed(state["messages"]):
        role = m.type if hasattr(m, "type") else m.get("role", "user")
        if role in ("human", "user"):
            return m.content if hasattr(m, "content") else m.get("content", "")
    return None


def format_messages(state: AgentState) -> list:
    """Convert state messages to Groq format."""
    formatted = []
    for m in state["messages"]:
        role = m.type if hasattr(m, "type") else m.get("role", "user")
        content = m.content if hasattr(m, "content") else m.get("content", "")
        if role in ("ai", "assistant"):
            try:
                content = json.loads(content).get("response", content)
            except Exception:
                pass
        if role in ("human", "user"):
            formatted.append({"role": "user", "content": content})
        elif role in ("ai", "assistant"):
            formatted.append({"role": "assistant", "content": content})
    return formatted

# ── Nodes ─────────────────────────────────────────────────────────────────────
def intent_classification_node(state: AgentState) -> AgentState:
    """Classify the newest employee message before its support stage runs."""
    message = state["messages"][-1]
    user_msg = message.content if hasattr(message, "content") else message.get("content", "")
    if user_msg.lower().strip() in TRIVIAL_MESSAGES:
        prediction = {"label": "normal", "confidence": 0.0}
    else:
        try:
            prediction = classify_employee_message(user_msg)
        except Exception:
            prediction = {"label": "normal", "confidence": 0.0, "degraded": True}
    return {**state, "intent_label": prediction["label"], "intent_confidence": prediction["confidence"],
            "intent_source_message": user_msg, "intent_degraded": prediction.get("degraded", False)}


def classifier_escalation(state: AgentState) -> tuple[bool, str, str, str, str, float]:
    """Translate the model intent into the existing escalation behavior.

    Always evaluates against the most recent HUMAN message rather than
    trusting state["intent_label"]/state["intent_confidence"] blindly. Those
    are only valid if they were computed for THIS message — e.g. qualify_node
    appends its own reply before summary_node's safety-net check runs, and a
    stage can in principle be entered without intent_classification_node
    having run first. When the cached label doesn't match the current human
    message (tracked via intent_source_message), reclassify on the spot
    instead of re-using a label left over from an earlier turn.
    """
    text = last_human_message(state)
    if text is None:
        return False, "", "", "", state.get("intent_label", "normal"), state.get("intent_confidence", 0.0)

    if state.get("intent_source_message") == text:
        label = state.get("intent_label", "normal")
        confidence = state.get("intent_confidence", 0.0)
    else:
        if text.lower().strip() in TRIVIAL_MESSAGES:
            prediction = {"label": "normal", "confidence": 0.0}
        else:
            try:
                prediction = classify_employee_message(text)
            except Exception:
                prediction = {"label": "normal", "confidence": 0.0}
        label, confidence = prediction["label"], prediction["confidence"]

    outcomes = {
        "frustrated": (
            "Frustrated sentiment detected by DistilBERT",
            "I'm sorry this has been frustrating. I'm going to connect you with one of our team members right away.",
        ),
        "explicit_human_request": (
            "Customer requested human agent",
            "I'm flagging this conversation for Human IT Support right away. An IT support analyst will follow up with you shortly.",
        ),
        "repeated_confusion": (
            "Repeated confusion detected by DistilBERT",
            "I can see the previous guidance has not resolved this. I'm escalating this to Human IT Support so they can take over.",
        ),
    }
    model_reason, model_response = outcomes.get(label, ("", ""))
    model_escalation = bool(model_reason) and confidence >= ESCALATION_CONFIDENCE_THRESHOLD

    is_negative, negative_reason = sentiment_check(text)
    is_explicit, explicit_reason = explicit_escalation(text)
    is_repeated_failure, repeated_failure_reason = repeated_failure_check(text)
    keyword_reason = negative_reason or explicit_reason or repeated_failure_reason
    keyword_escalation = is_negative or is_explicit or is_repeated_failure

    if model_escalation and keyword_escalation:
        return True, model_reason, model_response, "both", label, confidence
    if model_escalation:
        return True, model_reason, model_response, "distilbert", label, confidence
    if keyword_escalation:
        return True, keyword_reason, (
            "I'm sorry this has been frustrating. I'm going to connect you with one of our team members right away."
            if is_negative else
            "I'm flagging this conversation for Human IT Support right away. An IT support analyst will follow up with you shortly."
            if is_explicit else
            "I can see the previous guidance has not resolved this. I'm escalating this to Human IT Support so they can take over."
        ), "keyword", label, confidence
    return False, "", "", "", label, confidence


def escalation_result(state: AgentState):
    should_escalate, reason, response, trigger, label, confidence = classifier_escalation(state)
    if not should_escalate:
        return None

    qual = state.get("qualification", {})
    ticket = None
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
    log_escalation(state["session_id"], reason, format_messages(state), trigger, label, confidence)
    return {**state, "intent_label": label, "intent_confidence": confidence,
            "stage": "escalated", "escalation_reason": reason, "escalation_trigger": trigger,
            "messages": state["messages"] + [AIMessage(content=json.dumps(result))]}


# NEW: autonomous decision node ───────────────────────────────────────────────
def decision_node(state: AgentState) -> AgentState:
    """
    Decides which stage to route to next. When AUTONOMOUS_ROUTING is off, this
    is a no-op that preserves the original deterministic behavior (state["stage"]
    is left exactly as it was, so the existing router sends the request to the
    same node it always would have).

    When AUTONOMOUS_ROUTING is on:
      1. DistilBERT / keyword escalation signals are checked FIRST and act as a
         hard override — if they fire, we go straight to escalation, no LLM call.
      2. Otherwise an LLM call picks one of DECISION_ACTIONS.
      3. The LLM's output is validated against DECISION_ACTIONS; any invalid
         output or call failure falls back to the current deterministic stage.
      4. A repeat counter forces escalation if the same non-terminal action is
         chosen too many turns in a row (loop guard).
      5. Every outcome is logged to decision_log.json.
    """
    if not AUTONOMOUS_ROUTING:
        return state

    # 1. Hard override — DistilBERT / keyword escalation always wins.
    should_escalate, reason, _response, trigger, label, confidence = classifier_escalation(state)
    if should_escalate:
        log_decision(state["session_id"], "escalate", trigger, reason)
        return {**state, "intent_label": label, "intent_confidence": confidence,
                "stage": "escalated", "last_decision": "escalate", "decision_repeat_count": 0}

    # 2. Ask the LLM to pick the next action.
    qual_progress = f"{len(state.get('qualification', {}))}/{len(QUALIFY_QUESTIONS)}"
    state_summary = (
        f"Current deterministic stage (prior): {state['stage']}\n"
        f"Unanswered SOP questions in a row: {state.get('unanswered_count', 0)}\n"
        f"Qualification progress: {qual_progress}\n"
        f"SOP gaps logged so far: {len(state.get('sop_gaps', []))}\n"
    )
    decision_messages = format_messages(state) + [
        {"role": "user", "content": f"[STATE SUMMARY]\n{state_summary}\nChoose the next action."}
    ]

    action = None
    reasoning = ""
    overridden_by = ""
    try:
        raw = llm_call(DECISION_SYSTEM_PROMPT, decision_messages)
        candidate = str(raw.get("action", "")).strip().lower()
        reasoning = str(raw.get("reasoning", ""))
        if candidate in DECISION_ACTIONS:
            action = candidate
    except Exception:
        action = None

    # 3. Validation fallback — invalid output or call failure keeps the
    #    deterministic behavior instead of guessing.
    if action is None:
        fallback_map = {"faq": "faq", "qualify": "qualify", "escalated": "escalate", "summary": "summary"}
        action = fallback_map.get(state["stage"], "faq")
        overridden_by = "validation_fallback"
        reasoning = reasoning or "LLM decision invalid or unavailable; kept deterministic stage."

    # 4. Loop guard — same non-terminal action repeated too many times.
    last_decision = state.get("last_decision", "")
    repeat_count = state.get("decision_repeat_count", 0)
    if action == last_decision and action not in ("escalate", "summary"):
        repeat_count += 1
    else:
        repeat_count = 1

    if repeat_count >= MAX_REPEAT_DECISIONS and action not in ("escalate", "summary"):
        overridden_by = "turn_limit"
        reasoning = f"'{action}' repeated {repeat_count} times without progress; forcing escalation."
        action = "escalate"

    log_decision(state["session_id"], action, overridden_by, reasoning)

    stage_map = {"faq": "faq", "qualify": "qualify", "escalate": "escalated", "summary": "summary"}
    return {
        **state,
        "stage": stage_map[action],
        "last_decision": action,
        "decision_repeat_count": repeat_count,
    }


def faq_node(state: AgentState) -> AgentState:
    """Stage 1: Answer inbound questions from SOP only."""
    user_msg = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else state["messages"][-1].get("content", "")

    escalation = escalation_result(state)
    if escalation:
        return escalation

    result = assistant_payload(llm_call(SYSTEM_BASE, format_messages(state)))

    new_unanswered = state["unanswered_count"]
    new_gaps = state["sop_gaps"].copy()

    if result.get("sop_gap"):
        new_unanswered += 1
        new_gaps.append(user_msg)
    else:
        new_unanswered = 0

    if result.get("escalate") or result.get("sop_gap") or new_unanswered >= 2:
        reason = result.get("escalation_reason") or "Question outside SOP data"
        result["escalate"] = True
        result["escalation_reason"] = reason
        log_escalation(state["session_id"], reason, format_messages(state), "groq",
                       state["intent_label"], state["intent_confidence"])
        return {
            **state,
            "stage": "escalated",
            "escalation_reason": reason,
            "unanswered_count": new_unanswered,
            "sop_gaps": new_gaps,
            "messages": state["messages"] + [AIMessage(content=json.dumps(result))]
        }

    next_stage = "faq"
    qualification_started = state.get("qualification_started", False)
    if not result.get("sop_gap") and user_msg.lower().strip() not in {"hi", "hello", "hey", "hiya"}:
        next_stage = "qualify"
        qualification_started = True
        result["response"] = f"{result['response']} {QUALIFY_QUESTIONS[0]}"

    return {
        **state,
        "stage": next_stage,
        "unanswered_count": new_unanswered,
        "sop_gaps": new_gaps,
        "qualification_started": qualification_started,
        "messages": state["messages"] + [AIMessage(content=json.dumps(result))]
    }

def qualify_node(state: AgentState) -> AgentState:
    """Stage 2: Ask structured qualification questions."""
    user_msg = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else state["messages"][-1].get("content", "")

    escalation = escalation_result(state)
    if escalation:
        return escalation

    new_qual = state["qualification"].copy()
    q_count = len(new_qual)

    # Autonomous routing can send us straight into "qualify" before Question 1
    # has ever actually been asked (normally faq_node asks it as part of its
    # own reply before handing off). In that case the current message is the
    # user's ORIGINAL question, not an answer to anything — ask Question 1
    # instead of silently storing it as answer_1.
    if not state.get("qualification_started") and q_count == 0:
        result = assistant_payload({
            "response": QUALIFY_QUESTIONS[0],
            "confidence": 1.0,
            "stage_complete": False
        })
        return {
            **state,
            "stage": "qualify",
            "qualification_started": True,
            "messages": state["messages"] + [AIMessage(content=json.dumps(result))]
        }

    # A reply that just repeats the original question or an earlier answer
    # verbatim isn't actually answering the question just asked — accepting
    # it as-is lets qualification "complete" on copy-pasted non-answers and
    # rushes straight to a fabricated summary. Re-ask instead of advancing.
    normalized = user_msg.strip().lower()
    already_said = {a.strip().lower() for a in new_qual.values()} | {
        (m.content if hasattr(m, "content") else m.get("content", "")).strip().lower()
        for m in state["messages"][:-1]
        if (m.type if hasattr(m, "type") else m.get("role", "")) in ("human", "user")
    }
    if q_count < len(QUALIFY_QUESTIONS) and normalized in already_said:
        result = assistant_payload({
            "response": f"That looks like something you already told me rather than an answer to this question: {QUALIFY_QUESTIONS[q_count]}",
            "confidence": 1.0,
            "stage_complete": False
        })
        return {
            **state,
            "stage": "qualify",
            "messages": state["messages"] + [AIMessage(content=json.dumps(result))]
        }

    if q_count < len(QUALIFY_QUESTIONS):
        new_qual[f"answer_{q_count + 1}"] = user_msg

    if len(new_qual) >= len(QUALIFY_QUESTIONS):
        result = assistant_payload({
            "response": "Thank you, that helps. I have your employee information, issue details, and device or system details. I'll prepare a short session summary for Human IT Support.",
            "confidence": 1.0,
            "stage_complete": True
        })
        next_stage = "summary"
    else:
        result = assistant_payload({
            "response": QUALIFY_QUESTIONS[len(new_qual)],
            "confidence": 1.0,
            "stage_complete": False
        })
        next_stage = "qualify"

    return {
        **state,
        "stage": next_stage,
        "qualification": new_qual,
        "messages": state["messages"] + [AIMessage(content=json.dumps(result))]
    }

def escalation_node(state: AgentState) -> AgentState:
    """Stage 3: Handle escalation gracefully."""
    return {**state, "conversation_ended": True}

def summary_node(state: AgentState) -> AgentState:
    """Stage 4: Generate structured session summary."""
    # NEW: even summary is behind the DistilBERT hard gate when autonomous
    # routing sent us here directly — safety net in case qualification wasn't
    # actually complete (e.g. decision node jumped early).
    escalation = escalation_result(state)
    if escalation:
        return escalation

    history = format_messages(state)
    qual = state.get("qualification", {})
    gaps = state.get("sop_gaps", [])

    summary_prompt = f"""Based on this full conversation, generate a structured IT support session summary.

Qualification data collected: {json.dumps(qual)}
SOP gaps identified: {json.dumps(gaps)}

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

    groq_messages = [{"role": "system", "content": summary_prompt}] + history
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            reasoning_effort="low",
            messages=groq_messages
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
            summary = json.loads(raw)
        except Exception:
            summary = {"raw_summary": raw}
    except Exception as exc:
        # Groq call failed outright (e.g. a 400 parse-failure error) or came
        # back empty — fall back to a minimal safe summary built from what we
        # already collected, rather than crashing the conversation.
        print(f"[summary_node] Groq call failed, using fallback: {exc}")
        summary = {
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

    save_final_summary(state["session_id"], {
        "intent_label": state["intent_label"],
        "intent_confidence": state["intent_confidence"],
        "escalation_trigger": state["escalation_trigger"],
        **summary,
    })

    severity = summary.get("severity") or "medium"
    key_details = summary.get("key_details_collected") or {}
    ticket = None
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

    return {
        **state,
        "conversation_ended": True,
        "messages": state["messages"] + [AIMessage(content=json.dumps(result))]
    }

# ── Router ────────────────────────────────────────────────────────────────────
def router(state: AgentState) -> str:
    stage = state["stage"]
    if state.get("conversation_ended"):
        return END
    if stage == "faq":       return "faq"
    if stage == "qualify":   return "qualify"
    if stage == "escalated": return "escalation"
    if stage == "summary":   return "summary"
    return END

def after_qualify_router(state: AgentState) -> str:
    if state["stage"] == "summary":
        return "summary"
    if state["stage"] == "escalated":
        return "escalation"
    return END

# ── Build Graph ───────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("intent_classification", intent_classification_node)
    graph.add_node("decision", decision_node)  # NEW
    graph.add_node("faq", faq_node)
    graph.add_node("qualify", qualify_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("summary", summary_node)

    graph.set_conditional_entry_point(router, {
        "faq": "intent_classification",
        "qualify": "intent_classification",
        "escalation": "escalation",
        "summary": "summary",
        END: END
    })
    # NEW: intent_classification now always flows through decision before
    # dispatching. When AUTONOMOUS_ROUTING is false, decision_node is a no-op
    # passthrough, so behavior is byte-for-byte identical to before.
    graph.add_edge("intent_classification", "decision")
    graph.add_conditional_edges("decision", router, {
        "faq": "faq",
        "qualify": "qualify",
        "escalation": "escalation",
        "summary": "summary",
        END: END
    })
    graph.add_edge("faq", END)
    graph.add_conditional_edges("qualify", after_qualify_router, {
        "summary": "summary",
        "escalation": "escalation",
        END: END
    })
    graph.add_edge("escalation", END)
    graph.add_edge("summary", END)

    return graph.compile()

APP = build_graph()

# ── Helpers ───────────────────────────────────────────────────────────────────
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
        "intent_label": "normal",
        "intent_confidence": 0.0,
        "intent_source_message": "",
        "intent_degraded": False,
        "escalation_trigger": "",
        "last_decision": "",
        "decision_repeat_count": 0,
    }