"""FastAPI API for Closira. The browser UI is implemented in frontend/."""

import json
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage

from .agent import APP, get_initial_state
from .database import create_session, get_session, initialize_database, list_sessions, persist_session


@asynccontextmanager
async def lifespan(_app: FastAPI):
    initialize_database()
    yield

app = FastAPI(title="Closira IT Helpdesk AI", lifespan=lifespan)
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000").rstrip("/")
app.add_middleware(CORSMiddleware, allow_origins=[frontend_url], allow_credentials=True,
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


def session_payload(session: dict) -> dict:
    """Convert a stored session into the frontend's display format."""
    return {
        "session_id": str(session["session_id"]),
        "stage": session["stage"],
        "sop_gaps": session["sop_gaps"],
        "qualification": session["qualification"],
        "meta": {
            "predicted_escalation_label": session["intent_label"],
            "confidence": session["intent_confidence"],
            "escalation_reason": session["escalation_reason"],
        },
        "messages": [
            {
                "role": "user" if message["role"] == "user" else "ai",
                "text": message["content"],
                "confidence": message["metadata"].get("confidence"),
                "escalate": message["metadata"].get("escalate"),
                "label": message["metadata"].get("predicted_escalation_label", session["intent_label"]),
                "escalationReason": message["metadata"].get("escalation_reason"),
            }
            for message in session["messages"]
        ],
    }


def state_from_session(session: dict) -> dict:
    """Rebuild the existing LangGraph state without changing its workflow."""
    state = get_initial_state(str(session["session_id"]))
    state.update({
        "stage": session["stage"],
        "unanswered_count": session["unanswered_count"],
        "qualification": session["qualification"],
        "escalation_reason": session["escalation_reason"] or "",
        "sop_gaps": session["sop_gaps"],
        "conversation_ended": session["conversation_ended"],
        "intent_label": session["intent_label"],
        "intent_confidence": session["intent_confidence"],
        "escalation_trigger": session["escalation_trigger"] or "",
    })
    state["messages"] = [
        HumanMessage(content=message["content"])
        if message["role"] == "user"
        else AIMessage(content=json.dumps({"response": message["content"], **message["metadata"]}))
        for message in session["messages"]
    ]
    return state


def find_session(session_id: str | None) -> dict | None:
    """Avoid a database error when a browser has an invalid stale session ID."""
    try:
        uuid.UUID(session_id or "")
    except ValueError:
        return None
    return get_session(session_id)


@app.get("/sessions")
async def sessions():
    return [
        {**session, "session_id": str(session["session_id"])}
        for session in list_sessions()
    ]


@app.get("/sessions/{session_id}")
async def session_detail(session_id: str):
    session = find_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_payload(session)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    requested_session_id = websocket.query_params.get("session_id")
    stored_session = find_session(requested_session_id)
    if stored_session is not None:
        state = state_from_session(stored_session)
        session_id = state["session_id"]
    else:
        session_id = str(uuid.uuid4())
        state = get_initial_state(session_id)
        create_session(state)
    await websocket.send_json({"type": "session_id", "session_id": session_id, "message": WELCOME})
    if stored_session is not None:
        await websocket.send_json({"type": "session_state", **session_payload(stored_session)})
    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "").strip()
            if not user_message:
                continue
            state["messages"].append(HumanMessage(content=user_message))
            before = len(state["messages"])
            state = APP.invoke(state)
            persist_session(state)
            predicted_escalation_label = state.get("intent_label", "normal")
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
