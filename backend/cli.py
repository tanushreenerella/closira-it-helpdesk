"""
Closira IT Helpdesk CLI — Run a support conversation from the terminal.
Usage: python cli.py
"""

import os
import json
import sys
from langchain_core.messages import HumanMessage, AIMessage
from agent import (
    get_initial_state, SYSTEM_BASE, QUALIFY_PROMPT,
    llm_call, log_escalation, format_messages,
    sentiment_check, explicit_escalation
)

WELCOME = (
    "\n🛠️ Welcome to Closira IT Helpdesk AI Support Assistant\n"
    "   I can help with accounts, networks, VPN, devices, email, and software access.\n"
    "   Type 'quit' to end the session and generate a summary.\n"
    "─" * 55
)

STAGE_LABELS = {
    "faq":       "📋 IT Support",
    "qualify":   "✅ Issue Qualification",
    "escalated": "🔴 Escalated",
    "summary":   "📊 Summary"
}


def print_ai(text: str, meta: dict = None):
    print(f"\n  🛠️ IT Support: {text}")
    if meta and meta.get("confidence") is not None:
        conf = float(meta["confidence"])
        bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        emoji = "🟢" if conf >= 0.75 else "🟡" if conf >= 0.5 else "🔴"
        print(f"     {emoji} Confidence: [{bar}] {int(conf*100)}%")
    if meta and meta.get("escalate"):
        print(f"     ⚠️  ESCALATING — {meta.get('escalation_reason', '')}")
    if meta and meta.get("sop_gap"):
        print(f"     📭 SOP Gap: This question wasn't covered in our data")


def print_stage(stage: str):
    label = STAGE_LABELS.get(stage, stage)
    print(f"\n  ── Stage changed → {label} ──")


def generate_summary(state: dict) -> str:
    from agent import client, MODEL
    qual = state.get("qualification", {})
    gaps = state.get("sop_gaps", [])
    history = format_messages(state)

    summary_prompt = f"""Based on this full conversation, produce a structured session summary.
Qualification data: {json.dumps(qual)}
SOP gaps: {json.dumps(gaps)}

Return ONLY valid JSON:
{{
  "employee_intent": "<one sentence>",
  "key_details_collected": {{
    "employee": "<or null>",
    "issue": "<or null>",
    "device_system": "<or null>"
  }},
  "sop_gaps": ["<gap1>"],
  "recommended_next_action": "<what team should do>",
  "escalated": false,
  "escalation_reason": null
}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=summary_prompt,
        messages=history
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}


def main():
    import uuid
    session_id = str(uuid.uuid4())
    state = get_initial_state(session_id)
    prev_stage = "faq"

    print(WELCOME)
    print(f"  Session ID: {session_id[:8]}…\n")

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            user_input = "quit"

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye", "end"):
            print("\n  ── Generating session summary… ──")
            summary = generate_summary(state)
            print("\n" + "═" * 55)
            print("  📊 SESSION SUMMARY")
            print("═" * 55)
            for k, v in summary.items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for sk, sv in v.items():
                        print(f"    • {sk}: {sv}")
                elif isinstance(v, list):
                    print(f"  {k}: {', '.join(v) if v else 'None'}")
                else:
                    print(f"  {k}: {v}")
            print("═" * 55)
            print(f"\n  Summary saved. Thank you for using Closira IT Helpdesk. 🛠️\n")
            break

        # Add to state
        state["messages"].append(HumanMessage(content=user_input))

        # Pre-flight checks
        is_neg, neg_reason = sentiment_check(user_input)
        is_exp, exp_reason = explicit_escalation(user_input)

        if is_neg or is_exp:
            reason = neg_reason or exp_reason
            log_escalation(session_id, reason, format_messages(state))
            state["stage"] = "escalated"
            reply = ("I'm sorry this has been frustrating. I'm flagging this for Human IT Support "
                     "immediately — an IT support analyst will follow up shortly. 🛠️")
            print_ai(reply, {"escalate": True, "escalation_reason": reason, "confidence": 1.0})
            print(f"\n  🔴 ESCALATED — {reason}")
            print(f"  Logged to escalation_log.json\n")
            continue

        # Choose prompt
        system = SYSTEM_BASE
        if state["stage"] == "qualify":
            system = SYSTEM_BASE + "\n\n" + QUALIFY_PROMPT

        result = llm_call(system, format_messages(state))
        ai_text = result.get("response", "Sorry, something went wrong.")
        confidence = result.get("confidence", 0.7)
        escalate = result.get("escalate", False)
        esc_reason = result.get("escalation_reason")
        sop_gap = result.get("sop_gap", False)
        stage_complete = result.get("stage_complete", False)

        if sop_gap:
            state["unanswered_count"] += 1
            state["sop_gaps"].append(user_input)

        if state["unanswered_count"] >= 2 and not escalate:
            escalate = True
            esc_reason = "More than 2 consecutive unanswered questions"

        if escalate:
            log_escalation(session_id, esc_reason, format_messages(state))
            state["stage"] = "escalated"
        elif stage_complete and state["stage"] == "faq":
            state["stage"] = "qualify"
            print_stage("qualify")
        elif stage_complete and state["stage"] == "qualify":
            state["stage"] = "summary"
            print_stage("summary")

        if state["stage"] == "qualify":
            q_count = len(state["qualification"])
            if q_count < 3:
                state["qualification"][f"answer_{q_count + 1}"] = user_input

        state["messages"].append(AIMessage(content=ai_text))
        print_ai(ai_text, {"confidence": confidence, "escalate": escalate,
                            "escalation_reason": esc_reason, "sop_gap": sop_gap})

        if state["stage"] != prev_stage:
            prev_stage = state["stage"]

        if escalate:
            print(f"\n  Escalation logged to escalation_log.json")


if __name__ == "__main__":
    main()
