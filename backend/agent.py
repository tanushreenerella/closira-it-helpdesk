"""
Closira AI IT Helpdesk Support Agent
Built with LangGraph + Groq (LLaMA 3.3 70B)
"""

import json
import os
import uuid
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from groq import Groq
from dotenv import load_dotenv
try:
    from .nlp.classifier import classify_employee_message
    from .database import record_escalation, save_final_summary
except ImportError:
    from nlp.classifier import classify_employee_message
    from database import record_escalation, save_final_summary

load_dotenv()

# ── Load SOP ────────────────────────────────────────────────────────────────
with open(os.path.join(os.path.dirname(__file__), "sop.json"), "r") as f:
    SOP = json.load(f)

SOP_TEXT = json.dumps(SOP, indent=2)

# ── Groq client ──────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"

# ── Escalation log ───────────────────────────────────────────────────────────
ESCALATION_CONFIDENCE_THRESHOLD = 0.60
TRIVIAL_MESSAGES = {"hi", "hello", "hey", "hiya", "ok", "okay", "thanks", "thank you", "yes", "no", "yep", "nope"}

def log_escalation(session_id: str, reason: str, conversation: list, trigger: str = "groq",
                   intent_label: str = "normal", intent_confidence: float = 0.0):
    record_escalation(session_id, reason, trigger, intent_label, intent_confidence)

# ── State ────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: list
    session_id: str
    stage: Literal["faq", "qualify", "escalated", "summary"]
    unanswered_count: int
    qualification: dict
    escalation_reason: str
    sop_gaps: list
    conversation_ended: bool
    intent_label: str
    intent_confidence: float
    escalation_trigger: str

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
        "sop_gap": bool(result.get("sop_gap", False))
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
            prediction = {"label": "normal", "confidence": 0.0}
    return {**state, "intent_label": prediction["label"], "intent_confidence": prediction["confidence"]}


def classifier_escalation(state: AgentState, text: str) -> tuple[bool, str, str, str]:
    """Translate the model intent into the existing escalation behavior."""
    label = state.get("intent_label", "normal")
    confidence = state.get("intent_confidence", 0.0)
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
        return True, model_reason, model_response, "both"
    if model_escalation:
        return True, model_reason, model_response, "distilbert"
    if keyword_escalation:
        return True, keyword_reason, (
            "I'm sorry this has been frustrating. I'm going to connect you with one of our team members right away."
            if is_negative else
            "I'm flagging this conversation for Human IT Support right away. An IT support analyst will follow up with you shortly."
            if is_explicit else
            "I can see the previous guidance has not resolved this. I'm escalating this to Human IT Support so they can take over."
        ), "keyword"
    return False, "", "", ""


def escalation_result(state: AgentState):
    message = state["messages"][-1]
    text = message.content if hasattr(message, "content") else message.get("content", "")
    should_escalate, reason, response, trigger = classifier_escalation(state, text)
    if not should_escalate:
        return None
    result = assistant_payload({"response": response, "confidence": 1.0,
                                "escalate": True, "escalation_reason": reason})
    log_escalation(state["session_id"], reason, format_messages(state), trigger,
                   state["intent_label"], state["intent_confidence"])
    return {**state, "stage": "escalated", "escalation_reason": reason, "escalation_trigger": trigger,
            "messages": state["messages"] + [AIMessage(content=json.dumps(result))]}


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
    if not result.get("sop_gap") and user_msg.lower().strip() not in {"hi", "hello", "hey", "hiya"}:
        next_stage = "qualify"
        result["response"] = f"{result['response']} {QUALIFY_QUESTIONS[0]}"

    return {
        **state,
        "stage": next_stage,
        "unanswered_count": new_unanswered,
        "sop_gaps": new_gaps,
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
    history = format_messages(state)
    qual = state.get("qualification", {})
    gaps = state.get("sop_gaps", [])

    summary_prompt = f"""Based on this full conversation, generate a structured IT support session summary.

Qualification data collected: {json.dumps(qual)}
SOP gaps identified: {json.dumps(gaps)}

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
  "escalated": false,
  "escalation_reason": null
}}"""

    groq_messages = [{"role": "system", "content": summary_prompt}] + history
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
        summary = json.loads(raw)
    except Exception:
        summary = {"raw_summary": raw}

    save_final_summary(state["session_id"], {
        "intent_label": state["intent_label"],
        "intent_confidence": state["intent_confidence"],
        "escalation_trigger": state["escalation_trigger"],
        **summary,
    })

    result = assistant_payload({
        "response": "Session summary saved for Human IT Support. Thank you for using the AI Support Assistant.",
        "confidence": 1.0,
        "stage_complete": True
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
    graph.add_conditional_edges("intent_classification", router, {
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
        "escalation_reason": "",
        "sop_gaps": [],
        "conversation_ended": False,
        "intent_label": "normal",
        "intent_confidence": 0.0,
        "escalation_trigger": ""
    }
