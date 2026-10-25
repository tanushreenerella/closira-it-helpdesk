"""
Closira IT Helpdesk CLI — Run a support conversation from the terminal.
Usage: python -m backend.cli
"""

import json
import uuid

from backend.pipeline import generate_summary, get_initial_state, run_turn

WELCOME = (
    "\n🛠️ Welcome to Closira IT Helpdesk AI Support Assistant\n"
    "   I can help with accounts, networks, VPN, devices, email, and software access.\n"
    "   Type 'quit' to end the session and generate a summary.\n"
    + "─" * 55
)

STAGE_LABELS = {
    "faq":       "📋 IT Support",
    "qualify":   "✅ Issue Qualification",
    "troubleshoot": "🔧 Troubleshooting",
    "escalated": "🔴 Escalated",
    "summary":   "📊 Summary"
}


def print_ai(text: str, meta: dict = None):
    print(f"\n  🛠️ IT Support: {text}")
    if meta and meta.get("intent"):
        print(f"     🏷️  intent={meta['intent']} emotion={meta.get('emotion')} risk={meta.get('risk', 0):.2f}")
    if meta and meta.get("escalate"):
        print(f"     ⚠️  ESCALATING — {meta.get('escalation_reason', '')}")
    if meta and meta.get("sop_gap"):
        print(f"     📭 SOP Gap: This question wasn't covered in our data")


def print_summary(summary: dict):
    print("\n" + "═" * 55)
    print("  📊 SESSION SUMMARY")
    print("═" * 55)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    • {sk}: {sv}")
        elif isinstance(v, list):
            print(f"  {k}: {', '.join(map(str, v)) if v else 'None'}")
        else:
            print(f"  {k}: {v}")
    print("═" * 55)


def main():
    session_id = str(uuid.uuid4())
    state = get_initial_state(session_id)
    prev_stage = state["stage"]

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
            print_summary(generate_summary(state))
            print(f"\n  Thank you for using Closira IT Helpdesk. 🛠️\n")
            break

        before = len(state["messages"]) + 1
        state = run_turn(state, user_input)
        replies = [json.loads(m["content"]) for m in state["messages"][before:] if m["role"] == "assistant"]
        if not replies:
            print("\n  This conversation has already been handed off. Type 'quit' to finish.")
        for reply in replies:
            print_ai(reply["response"], reply)

        if state["stage"] != prev_stage:
            print(f"\n  ── Stage changed → {STAGE_LABELS.get(state['stage'], state['stage'])} ──")
            prev_stage = state["stage"]


if __name__ == "__main__":
    main()
