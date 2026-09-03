"""
Closira Web Server — FastAPI + WebSocket streaming
LLM: Groq (LLaMA 3.3 70B) — free tier, no card required
"""

import json
import os
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessage

from agent import (
    get_initial_state, APP, SYSTEM_BASE, QUALIFY_PROMPT,
    llm_call, log_escalation, format_messages, sentiment_check,
    explicit_escalation, SOP
)

app = FastAPI(title="Closira IT Helpdesk AI")

# ── In-memory session store ──────────────────────────────────────────────────
sessions: dict = {}

WELCOME = (
    "Hi there! Welcome to Closira IT Helpdesk. I'm your AI Support Assistant. "
    "I can help with accounts, Wi-Fi and network, VPN, devices, email, and software access. "
    "How can I help you today?"
)

def parse_agent_message(message) -> dict:
    """Read the structured JSON payload produced by LangGraph nodes."""
    content = message.content if hasattr(message, "content") else message.get("content", "")
    try:
        payload = json.loads(content)
    except Exception:
        payload = {"response": str(content), "confidence": 0.7}

    response = str(payload.get("response", "")).strip()
    if "{" in response:
        response = response[:response.index("{")].strip()

    return {
        "response": response or "I'm sorry, I didn't quite catch that. Could you rephrase?",
        "confidence": payload.get("confidence", 0.7),
        "escalate": payload.get("escalate", False),
        "escalation_reason": payload.get("escalation_reason"),
        "sop_gap": payload.get("sop_gap", False),
        "stage_complete": payload.get("stage_complete", False),
    }

# ── HTML UI ──────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Closira IT Helpdesk — AI Support Assistant</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=Inter:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
  :root {
    --rose:       #c9848a;
    --rose-light: #f5e8e9;
    --rose-dark:  #9e5f64;
    --gold:       #c9a96e;
    --cream:      #fdf8f5;
    --charcoal:   #2d2d2d;
    --muted:      #8a8a8a;
    --border:     #ede0d8;
    --white:      #ffffff;
    --escalate:   #e07b54;
    --shadow:     0 4px 24px rgba(180,120,110,0.10);
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', sans-serif;
    background: var(--cream);
    color: var(--charcoal);
    height: 100dvh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── Header ── */
  header {
    background: var(--white);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-shrink: 0;
    box-shadow: var(--shadow);
    z-index: 10;
  }
  .logo-circle {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, var(--rose), var(--rose-dark));
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
  }
  .header-info h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--charcoal);
    letter-spacing: 0.02em;
  }
  .header-info p {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 1px;
    letter-spacing: 0.03em;
  }
  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6bbf7a;
    margin-left: auto;
    box-shadow: 0 0 0 3px rgba(107,191,122,0.2);
    animation: pulse-green 2s infinite;
    flex-shrink: 0;
  }
  @keyframes pulse-green {
    0%,100% { box-shadow: 0 0 0 3px rgba(107,191,122,0.2); }
    50%      { box-shadow: 0 0 0 6px rgba(107,191,122,0.08); }
  }

  /* ── Stage banner ── */
  #stage-banner {
    background: var(--rose-light);
    border-bottom: 1px solid var(--border);
    padding: 7px 24px;
    font-size: 0.72rem;
    color: var(--rose-dark);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 500;
    flex-shrink: 0;
    transition: background 0.3s;
  }
  #stage-banner.escalated { background: #fdecea; color: var(--escalate); }
  #stage-banner.qualify   { background: #eef5fb; color: #4a7fa5; }
  #stage-banner.summary   { background: #eafaf1; color: #3a8c5a; }

  /* ── Main layout ── */
  .main {
    display: flex;
    flex: 1;
    overflow: hidden;
    gap: 0;
  }

  /* ── Chat panel ── */
  .chat-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border-right: 1px solid var(--border);
  }
  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    scroll-behavior: smooth;
  }
  #messages::-webkit-scrollbar { width: 4px; }
  #messages::-webkit-scrollbar-track { background: transparent; }
  #messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .msg {
    display: flex;
    gap: 10px;
    animation: fadeUp 0.25s ease both;
    max-width: 100%;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .msg.user  { flex-direction: row-reverse; }
  .msg.system { justify-content: center; }

  .avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .msg.ai   .avatar { background: linear-gradient(135deg, var(--rose), var(--rose-dark)); }
  .msg.user .avatar { background: linear-gradient(135deg, #a0b4c8, #7090a8); }

  .bubble {
    padding: 11px 15px;
    border-radius: 16px;
    font-size: 0.875rem;
    line-height: 1.55;
    max-width: min(72%, 520px);
    word-break: break-word;
  }
  .msg.ai   .bubble {
    background: var(--white);
    border: 1px solid var(--border);
    border-top-left-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }
  .msg.user .bubble {
    background: linear-gradient(135deg, var(--rose), var(--rose-dark));
    color: var(--white);
    border-top-right-radius: 4px;
  }
  .msg.system .bubble {
    background: transparent;
    color: var(--muted);
    font-size: 0.75rem;
    text-align: center;
    font-style: italic;
    border: none;
    padding: 4px 12px;
  }
  .escalate-bubble {
    background: #fff5f3 !important;
    border-color: #f5c3b8 !important;
    color: var(--escalate) !important;
  }
  .confidence-tag {
    display: inline-block;
    font-size: 0.65rem;
    padding: 2px 7px;
    border-radius: 20px;
    margin-top: 6px;
    font-weight: 500;
    letter-spacing: 0.04em;
  }
  .conf-high { background: #eafaf1; color: #3a8c5a; }
  .conf-mid  { background: #fef9e7; color: #b7950b; }
  .conf-low  { background: #fdecea; color: var(--escalate); }

  /* typing indicator */
  .typing-indicator { display: flex; gap: 5px; padding: 14px 16px; }
  .typing-indicator span {
    width: 7px; height: 7px;
    background: var(--rose);
    border-radius: 50%;
    animation: bounce 1.2s infinite;
    opacity: 0.6;
  }
  .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
  .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce {
    0%,80%,100% { transform: translateY(0); }
    40%          { transform: translateY(-6px); }
  }

  /* ── Input bar ── */
  .input-bar {
    padding: 16px 20px;
    background: var(--white);
    border-top: 1px solid var(--border);
    display: flex;
    gap: 10px;
    align-items: flex-end;
    flex-shrink: 0;
  }
  #user-input {
    flex: 1;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 0.875rem;
    font-family: 'Inter', sans-serif;
    color: var(--charcoal);
    background: var(--cream);
    resize: none;
    max-height: 120px;
    min-height: 42px;
    outline: none;
    transition: border-color 0.2s;
    line-height: 1.5;
  }
  #user-input:focus { border-color: var(--rose); }
  #user-input::placeholder { color: var(--muted); }
  #send-btn {
    width: 42px; height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--rose), var(--rose-dark));
    border: none;
    color: white;
    font-size: 16px;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    transition: opacity 0.2s, transform 0.15s;
  }
  #send-btn:hover { opacity: 0.9; transform: scale(1.04); }
  #send-btn:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }

  /* ── Side panel ── */
  .side-panel {
    width: 410px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    background: var(--white);
    padding: 18px 24px;
    gap: 14px;
  }
  .side-panel::-webkit-scrollbar { width: 4px; }
  .side-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .panel-card {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    background: var(--white);
  }
  .panel-card-header {
    background: var(--rose-light);
    padding: 9px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--rose-dark);
    display: flex;
    align-items: center;
    gap: 6px;
    line-height: 1.25;
  }
  .panel-card-body {
    padding: 12px 16px;
    line-height: 1.45;
    overflow-wrap: anywhere;
  }
  .panel-card-body p {
    font-size: 0.78rem;
    line-height: 1.45;
    color: var(--muted);
    font-style: italic;
  }
  #qual-body,
  #gaps-body {
    max-height: 128px;
    overflow-y: auto;
  }
  #qual-body::-webkit-scrollbar,
  #gaps-body::-webkit-scrollbar { width: 4px; }
  #qual-body::-webkit-scrollbar-thumb,
  #gaps-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  .panel-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 14px;
    min-height: 26px;
    margin-bottom: 9px;
    font-size: 0.8rem;
  }
  .panel-row:last-child { margin-bottom: 0; }
  .panel-label {
    color: var(--muted);
    flex: 0 0 92px;
  }
  .panel-value {
    color: var(--charcoal);
    text-align: right;
    font-weight: 500;
    min-width: 0;
    overflow-wrap: anywhere;
  }

  .sop-gap-item {
    font-size: 0.78rem;
    color: var(--escalate);
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
    line-height: 1.45;
    overflow-wrap: anywhere;
  }
  .sop-gap-item:last-child { border-bottom: none; }

  .qual-answer {
    font-size: 0.78rem;
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
    line-height: 1.45;
    overflow-wrap: anywhere;
  }
  .qual-answer:last-child { border-bottom: none; }
  .qual-q { color: var(--muted); font-size: 0.71rem; margin-bottom: 2px; }

  .stage-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .session-id-value {
    color: var(--muted);
    font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
    font-size: 0.72rem;
    line-height: 1.4;
  }
  .pill-faq      { background: var(--rose-light); color: var(--rose-dark); }
  .pill-qualify  { background: #eef5fb; color: #4a7fa5; }
  .pill-escalated{ background: #fdecea; color: var(--escalate); }
  .pill-summary  { background: #eafaf1; color: #3a8c5a; }

  .new-session-btn {
    width: 100%;
    padding: 10px;
    border-radius: 10px;
    background: transparent;
    border: 1.5px solid var(--rose);
    color: var(--rose-dark);
    font-size: 0.8rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s, color 0.2s;
    font-family: 'Inter', sans-serif;
  }
  .new-session-btn:hover { background: var(--rose-light); }

  /* ── Responsive ── */
  @media (max-width: 680px) {
    .side-panel { display: none; }
    .chat-panel { border-right: none; }
    #stage-banner { font-size: 0.68rem; padding: 6px 16px; }
    #messages { padding: 16px 12px; }
    .input-bar { padding: 12px; }
    .bubble { max-width: 85%; }
  }
  @media (max-width: 980px) {
    .side-panel { width: 340px; padding: 16px 14px; }
    .panel-card-body { padding: 12px; }
    .panel-row { gap: 10px; }
    .panel-label { flex-basis: 84px; }
  }
</style>
</head>
<body>

<header>
  <div class="logo-circle">🌸</div>
  <div class="header-info">
    <h1>Closira IT Helpdesk</h1>
    <p>AI Support Assistant — Powered by Groq Llama 3.3</p>
  </div>
  <div class="status-dot" title="Online"></div>
</header>

<div id="stage-banner">Stage: IT Support — Answering your questions</div>

<div class="main">
  <!-- Chat -->
  <div class="chat-panel">
    <div id="messages"></div>
    <div class="input-bar">
      <textarea id="user-input" placeholder="Describe your IT issue…" rows="1"></textarea>
      <button id="send-btn" title="Send">➤</button>
    </div>
  </div>

  <!-- Side panel -->
  <aside class="side-panel">

    <div class="panel-card">
      <div class="panel-card-header">📊 Session Info</div>
      <div class="panel-card-body">
        <div class="panel-row">
          <span class="panel-label">Stage</span>
          <span class="panel-value"><span id="stage-pill" class="stage-pill pill-faq">FAQ</span></span>
        </div>
        <div class="panel-row">
          <span class="panel-label">Confidence</span>
          <span class="panel-value" id="conf-val">—</span>
        </div>
        <div class="panel-row">
          <span class="panel-label">SOP Gaps</span>
          <span class="panel-value" id="gap-count">0</span>
        </div>
        <div class="panel-row">
          <span class="panel-label">Session ID</span>
          <span class="panel-value session-id-value" id="session-id-disp">—</span>
        </div>
      </div>
    </div>

    <div class="panel-card">
      <div class="panel-card-header">✅ Issue Qualification</div>
      <div class="panel-card-body" id="qual-body">
        <p>Employee, issue, and device details will appear here.</p>
      </div>
    </div>

    <div class="panel-card">
      <div class="panel-card-header">⚠️ SOP Gaps Detected</div>
      <div class="panel-card-body" id="gaps-body">
        <p>No gaps yet.</p>
      </div>
    </div>

    <button class="new-session-btn" onclick="newSession()">+ New Session</button>
  </aside>
</div>

<script>
let ws;
let sessionId = null;

const QUAL_LABELS = [
  "Treatment interest",
  "Prior experience",
  "Booking readiness"
];

function newSession() {
  if (ws) ws.close();
  document.getElementById('messages').innerHTML = '';
  document.getElementById('qual-body').innerHTML = '<p>Will populate during qualification stage.</p>';
  document.getElementById('gaps-body').innerHTML = '<p>No gaps yet.</p>';
  document.getElementById('conf-val').textContent = '—';
  document.getElementById('gap-count').textContent = '0';
  document.getElementById('session-id-disp').textContent = '—';
  document.getElementById('session-id-disp').removeAttribute('title');
  updateStage('faq');
  connect();
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    document.getElementById('send-btn').disabled = false;
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    removeTyping();

    if (data.type === 'session_id') {
      sessionId = data.session_id;
      const sessionDisplay = document.getElementById('session-id-disp');
      sessionDisplay.textContent = sessionId.slice(0, 12) + '…';
      sessionDisplay.title = sessionId;
      appendMessage('ai', data.message, null, false);
      return;
    }

    if (data.type === 'message') {
      const isEscalate = data.meta?.escalate;
      appendMessage('ai', data.message, data.meta, isEscalate);
      updateMeta(data.meta, data.stage, data.sop_gaps, data.qualification);
    }

    if (data.type === 'system') {
      appendMessage('system', data.message, null, false);
    }
  };

  ws.onclose = () => {
    document.getElementById('send-btn').disabled = true;
  };
}

function appendMessage(role, text, meta, isEscalate) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'ai' ? '🌸' : (role === 'user' ? '👤' : '');

  const bubble = document.createElement('div');
  bubble.className = 'bubble' + (isEscalate ? ' escalate-bubble' : '');
  bubble.textContent = text;

  if (meta && meta.confidence != null && role === 'ai') {
    const conf = parseFloat(meta.confidence);
    const tag = document.createElement('div');
    tag.className = 'confidence-tag ' + (conf >= 0.75 ? 'conf-high' : conf >= 0.5 ? 'conf-mid' : 'conf-low');
    tag.textContent = `Confidence: ${Math.round(conf*100)}%`;
    bubble.appendChild(tag);
  }

  if (role !== 'system') wrap.appendChild(avatar);
  wrap.appendChild(bubble);

  document.getElementById('messages').appendChild(wrap);
  document.getElementById('messages').scrollTop = 999999;
}

function showTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'msg ai';
  wrap.id = 'typing';
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = '🌸';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  document.getElementById('messages').appendChild(wrap);
  document.getElementById('messages').scrollTop = 999999;
}

function removeTyping() {
  const t = document.getElementById('typing');
  if (t) t.remove();
}

function updateStage(stage) {
  const banner = document.getElementById('stage-banner');
  const pill   = document.getElementById('stage-pill');
  const labels = {
    faq:       ['FAQ — Answering your questions', 'pill-faq',       'FAQ'],
    qualify:   ['Lead Qualification in progress', 'pill-qualify',   'Qualifying'],
    escalated: ['Escalated — Handing off to team', 'pill-escalated','Escalated'],
    summary:   ['Session complete — Summary generated', 'pill-summary', 'Summary']
  };
  const [bannerText, pillClass, pillText] = labels[stage] || labels['faq'];
  banner.textContent = 'Stage: ' + bannerText;
  banner.className = stage;
  pill.textContent = pillText;
  pill.className = 'stage-pill ' + pillClass;
  document.getElementById('stage-pill').textContent = pillText;
}

function updateMeta(meta, stage, sop_gaps, qualification) {
  if (stage) updateStage(stage);

  if (meta?.confidence != null) {
    const pct = Math.round(parseFloat(meta.confidence) * 100);
    document.getElementById('conf-val').textContent = pct + '%';
  }

  if (sop_gaps != null) {
    document.getElementById('gap-count').textContent = sop_gaps.length;
    const gapBody = document.getElementById('gaps-body');
    if (sop_gaps.length === 0) {
      gapBody.innerHTML = '<p>No gaps yet.</p>';
    } else {
      gapBody.innerHTML = sop_gaps.map(g =>
        `<div class="sop-gap-item">❓ ${g}</div>`
      ).join('');
    }
  }

  if (qualification && Object.keys(qualification).length > 0) {
    const qualBody = document.getElementById('qual-body');
    qualBody.innerHTML = Object.entries(qualification).map(([k, v], i) =>
      `<div class="qual-answer"><div class="qual-q">${QUAL_LABELS[i] || k}</div>${v}</div>`
    ).join('');
  }
}

function send() {
  const input = document.getElementById('user-input');
  const text = input.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  appendMessage('user', text, null, false);
  showTyping();
  ws.send(JSON.stringify({ message: text }));
  input.value = '';
  input.style.height = 'auto';
}

document.getElementById('send-btn').addEventListener('click', send);
document.getElementById('user-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
document.getElementById('user-input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

connect();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    state = get_initial_state(session_id)

    await websocket.send_json({
        "type": "session_id",
        "session_id": session_id,
        "message": WELCOME
    })

    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "").strip()
            if not user_message:
                continue

            state["messages"].append(HumanMessage(content=user_message))
            message_count_before_graph = len(state["messages"])

            state = APP.invoke(state)
            new_agent_messages = [
                msg for msg in state["messages"][message_count_before_graph:]
                if getattr(msg, "type", None) == "ai" or getattr(msg, "role", None) == "assistant"
            ]

            if not new_agent_messages:
                await websocket.send_json({
                    "type": "system",
                    "message": "This conversation has already reached a handoff stage. Please start a new session to continue testing."
                })
                continue

            for msg in new_agent_messages:
                result = parse_agent_message(msg)
                await websocket.send_json({
                    "type": "message",
                    "message": result["response"],
                    "stage": state["stage"],
                    "sop_gaps": state["sop_gaps"],
                    "qualification": state["qualification"],
                    "meta": {
                        "confidence": result["confidence"],
                        "escalate": result["escalate"],
                        "escalation_reason": result["escalation_reason"],
                        "sop_gap": result["sop_gap"],
                        "stage_complete": result["stage_complete"]
                    }
                })

                if result["escalate"]:
                    await websocket.send_json({
                        "type": "system",
                        "message": f"🔴 Escalated to human team — Reason: {result['escalation_reason']}"
                    })

            if state["conversation_ended"] and state["stage"] == "summary":
                await websocket.send_json({
                    "type": "system",
                    "message": f"📋 Summary saved as summary_{session_id[:8]}.json"
                })

            continue

            # Pre-flight: sentiment + explicit escalation
            is_neg, neg_reason = sentiment_check(user_message)
            is_exp, exp_reason = explicit_escalation(user_message)

            if is_neg or is_exp:
                reason = neg_reason or exp_reason
                log_escalation(session_id, reason, format_messages(state))
                state["stage"] = "escalated"
                state["escalation_reason"] = reason
                reply = (
                    "I'm really sorry to hear that. I'm flagging this conversation "
                    "for Human IT Support right away — an IT support analyst will "
                    "follow up with you shortly. 🛠️"
                )
                state["messages"].append(AIMessage(content=reply))
                await websocket.send_json({
                    "type": "message",
                    "message": reply,
                    "stage": "escalated",
                    "sop_gaps": state["sop_gaps"],
                    "qualification": state["qualification"],
                    "meta": {"escalate": True, "escalation_reason": reason, "confidence": 1.0}
                })
                await websocket.send_json({"type": "system", "message": f"🔴 Escalated: {reason}"})
                continue

            # Choose system prompt
            system = SYSTEM_BASE
            if state["stage"] == "qualify":
                system = SYSTEM_BASE + "\n\n" + QUALIFY_PROMPT

            # Legacy manual path (inactive): call LLM
            msgs = format_messages(state)
            result = llm_call(system, msgs)

            ai_text = result.get("response", "I'm sorry, I didn't quite catch that. Could you rephrase?")
            # Strip any trailing JSON blob Groq may append to the response text
            if "{" in ai_text:
                ai_text = ai_text[:ai_text.index("{")].strip()
            confidence = result.get("confidence", 0.7)
            escalate = result.get("escalate", False)
            esc_reason = result.get("escalation_reason")
            sop_gap = result.get("sop_gap", False)
            stage_complete = result.get("stage_complete", False)

            # SOP gap tracking
            if sop_gap:
                state["unanswered_count"] += 1
                state["sop_gaps"].append(user_message)

            # Threshold escalation
            if state["unanswered_count"] >= 2 and not escalate:
                escalate = True
                esc_reason = "More than 2 consecutive unanswered questions"

            if escalate:
                log_escalation(session_id, esc_reason, format_messages(state))
                state["stage"] = "escalated"
                state["escalation_reason"] = esc_reason

            elif stage_complete and state["stage"] == "faq":
                state["stage"] = "qualify"

            elif stage_complete and state["stage"] == "qualify":
                state["stage"] = "summary"
                # Generate summary asynchronously (fire-and-forget via message)
                await websocket.send_json({
                    "type": "system",
                    "message": "📋 Generating session summary…"
                })

            # Qualification tracking
            if state["stage"] == "qualify" or (stage_complete and state.get("_prev_stage") == "qualify"):
                q_count = len(state["qualification"])
                if q_count < 3:
                    state["qualification"][f"answer_{q_count + 1}"] = user_message

            state["_prev_stage"] = state["stage"]
            state["messages"].append(AIMessage(content=ai_text))

            await websocket.send_json({
                "type": "message",
                "message": ai_text,
                "stage": state["stage"],
                "sop_gaps": state["sop_gaps"],
                "qualification": state["qualification"],
                "meta": {
                    "confidence": confidence,
                    "escalate": escalate,
                    "escalation_reason": esc_reason,
                    "sop_gap": sop_gap
                }
            })

            if escalate:
                await websocket.send_json({
                    "type": "system",
                    "message": f"🔴 Escalated to human team — Reason: {esc_reason}"
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "system", "message": f"Connection error: {str(e)}"})
        except Exception:
            pass
