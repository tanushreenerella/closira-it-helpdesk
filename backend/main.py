"""FastAPI API for Closira. The browser UI is implemented in frontend/."""

import json
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

from .agent import APP, get_initial_state
from .nlp.classify import classify

app = FastAPI(title="Closira IT Helpdesk AI")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

WELCOME = ("Hi there! Welcome to Closira IT Helpdesk. I'm your AI Support Assistant. "
           "I can help with accounts, Wi-Fi and network, VPN, devices, email, and software access. How can I help you today?")


def parse_agent_message(message) -> dict:
    content = message.content if hasattr(message, "content") else message.get("content", "")
    try:
        payload = json.loads(content)
    except Exception:
        payload = {"response": str(content), "confidence": 0.7}
    response = str(payload.get("response", "")).strip().split("{", 1)[0].strip()
    return {"response": response or "I'm sorry, I didn't quite catch that. Could you rephrase?",
            "confidence": payload.get("confidence", 0.7), "escalate": payload.get("escalate", False),
            "escalation_reason": payload.get("escalation_reason"), "sop_gap": payload.get("sop_gap", False),
            "stage_complete": payload.get("stage_complete", False)}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    state = get_initial_state(session_id)
    await websocket.send_json({"type": "session_id", "session_id": session_id, "message": WELCOME})
    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "").strip()
            if not user_message:
                continue
            predicted_escalation_label = classify(user_message)
            state["messages"].append(HumanMessage(content=user_message))
            before = len(state["messages"])
            state = APP.invoke(state)
            agent_messages = [m for m in state["messages"][before:]
                              if getattr(m, "type", None) == "ai" or getattr(m, "role", None) == "assistant"]
            if not agent_messages:
                await websocket.send_json({"type": "system", "message": "This conversation has already reached a handoff stage. Please start a new session to continue testing."})
                continue
            for message in agent_messages:
                result = parse_agent_message(message)
                await websocket.send_json({
                    "type": "message", "message": result["response"], "stage": state["stage"],
                    "sop_gaps": state["sop_gaps"], "qualification": state["qualification"],
                    "meta": {**result, "predicted_escalation_label": predicted_escalation_label},
                })
                if result["escalate"]:
                    await websocket.send_json({"type": "system", "message": f"🔴 Escalated to human team — Reason: {result['escalation_reason']}"})
            if state["conversation_ended"] and state["stage"] == "summary":
                await websocket.send_json({"type": "system", "message": f"📋 Summary saved as summary_{session_id[:8]}.json"})
    except WebSocketDisconnect:
        pass
    except Exception as error:
        try:
            await websocket.send_json({"type": "system", "message": f"Connection error: {error}"})
        except Exception:
            pass
