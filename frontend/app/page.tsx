"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";

type Message = { role: "ai" | "user" | "system"; text: string; confidence?: number; escalate?: boolean; label?: string; escalationReason?: string };
type Meta = { confidence?: number; predicted_escalation_label?: string; escalation_reason?: string };
type SessionSummary = { session_id: string; stage: string; preview: string; updated_at: string };
type SessionState = { session_id: string; stage: string; sop_gaps: string[]; qualification: Record<string, string>; meta: Meta; messages: Message[] };

const labels: Record<string, string> = { faq: "AI support", qualify: "Information gathering", escalated: "Human support required", summary: "Session summary" };
const categories = ["Account & Password", "Wi-Fi / Network", "VPN", "Device Issues", "Email", "Software & Access"];
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";
const API_URL = process.env.NEXT_PUBLIC_API_URL || WS_URL.replace(/^ws/, "http").replace(/\/ws$/, "");
const SESSION_STORAGE_KEY = "closira.session_id";

function confidenceTone(value = 0) { return value >= .75 ? "bg-emerald-50 text-emerald-800 ring-emerald-200" : value >= .5 ? "bg-amber-50 text-amber-800 ring-amber-200" : "bg-red-50 text-red-800 ring-red-200"; }
function stageTone(stage: string) { return stage === "escalated" ? "bg-red-50 text-red-800 ring-red-200" : stage === "summary" ? "bg-emerald-50 text-emerald-800 ring-emerald-200" : stage === "qualify" ? "bg-amber-50 text-amber-800 ring-amber-200" : "bg-teal-50 text-teal-800 ring-teal-200"; }
function Icon({ name }: { name: string }) { const icons: Record<string, string> = { shield: "◈", spark: "✦", user: "●", send: "↑", plus: "+", device: "▣" }; return <span aria-hidden="true">{icons[name]}</span>; }
function Card({ title, children, warning = false }: { title: string; children: React.ReactNode; warning?: boolean }) { return <section className={`overflow-hidden rounded-xl border bg-white shadow-sm ${warning ? "border-red-200" : "border-slate-200"}`}><h2 className={`border-b px-4 py-3 text-[11px] font-semibold uppercase tracking-[.12em] ${warning ? "border-red-100 bg-red-50 text-red-800" : "border-slate-100 bg-slate-50 text-slate-600"}`}>{title}</h2><div className="p-4">{children}</div></section>; }

export default function Home() {
  const ws = useRef<WebSocket | null>(null);
  const messagesEnd = useRef<HTMLDivElement>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [text, setText] = useState("");
  const [stage, setStage] = useState("faq");
  const [session, setSession] = useState("");
  const [gaps, setGaps] = useState<string[]>([]);
  const [qualification, setQualification] = useState<Record<string, string>>({});
  const [meta, setMeta] = useState<Meta>({});
  const [connected, setConnected] = useState(false);
  const [history, setHistory] = useState<SessionSummary[]>([]);

  const loadHistory = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/sessions`);
      if (response.ok) setHistory(await response.json());
    } catch {
      // The active WebSocket still reports backend connection failures.
    }
  }, []);

  const applySessionState = useCallback((saved: SessionState) => {
    setSession(saved.session_id);
    setMessages(saved.messages);
    setStage(saved.stage);
    setGaps(saved.sop_gaps || []);
    setQualification(saved.qualification || {});
    setMeta(saved.meta || {});
  }, []);

  const connect = useCallback((requestedSessionId?: string, preserveDisplay = false) => {
    // Safely tear down any existing socket before opening a new one.
    // Nulling onclose first prevents a stale handler from firing during
    // teardown and touching state on an unmounting/reconnecting component
    // — this is what was causing "destroy is not a function" under
    // Fast Refresh.
    if (ws.current) {
      ws.current.onclose = null;
      ws.current.onmessage = null;
      ws.current.onopen = null;
      ws.current.close();
      ws.current = null;
    }

    if (!preserveDisplay) {
      setMessages([]);
      setStage("faq");
      setSession("");
      setGaps([]);
      setQualification({});
      setMeta({});
    }

    const activeSessionId = requestedSessionId || window.localStorage.getItem(SESSION_STORAGE_KEY);
    const socket = new WebSocket(activeSessionId ? `${WS_URL}?session_id=${encodeURIComponent(activeSessionId)}` : WS_URL);
    ws.current = socket;

    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onmessage = ({ data }) => {
      const payload = JSON.parse(data);
      if (payload.type === "session_id") {
        setSession(payload.session_id);
        window.localStorage.setItem(SESSION_STORAGE_KEY, payload.session_id);
        void loadHistory();
      } else if (payload.type === "session_state") {
        applySessionState(payload as SessionState);
      } else if (payload.type === "message") {
        setMessages((old) => [
          ...old,
          {
            role: "ai",
            text: payload.message,
            confidence: payload.meta?.confidence,
            escalate: payload.meta?.escalate,
            label: payload.meta?.predicted_escalation_label,
            escalationReason: payload.meta?.escalation_reason,
          },
        ]);
        setStage(payload.stage);
        setGaps(payload.sop_gaps || []);
        setQualification(payload.qualification || {});
        setMeta(payload.meta || {});
      } else if (payload.type === "system") {
        setMessages((old) => [...old, { role: "system", text: payload.message }]);
      }
    };
  }, [applySessionState, loadHistory]);

  const openSession = useCallback(async (sessionId: string) => {
    try {
      const response = await fetch(`${API_URL}/sessions/${encodeURIComponent(sessionId)}`);
      if (!response.ok) return;
      const saved = await response.json() as SessionState;
      window.localStorage.setItem(SESSION_STORAGE_KEY, saved.session_id);
      applySessionState(saved);
      connect(saved.session_id, true);
    } catch {
      // Leave the current chat untouched if the saved session cannot be loaded.
    }
  }, [applySessionState, connect]);

  const startNewSession = useCallback(() => {
    window.localStorage.removeItem(SESSION_STORAGE_KEY);
    connect();
  }, [connect]);

  useEffect(() => {
    connect();
    void loadHistory();

    return () => {
      if (ws.current) {
        ws.current.onclose = null;
        ws.current.onmessage = null;
        ws.current.onopen = null;
        ws.current.close();
        ws.current = null;
      }
    };
  }, [connect, loadHistory]);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = (event: FormEvent) => {
    event.preventDefault();
    const value = text.trim();
    if (!value || ws.current?.readyState !== WebSocket.OPEN) return;
    setMessages((old) => [...old, { role: "user", text: value }]);
    ws.current.send(JSON.stringify({ message: value }));
    setText("");
  };

  const escalationReason =
    meta.escalation_reason ||
    [...messages].reverse().find((message) => message.escalationReason)?.escalationReason;

  return (
    <main className="flex h-[100dvh] flex-col overflow-hidden bg-[#f8faf9] font-sans text-slate-800">
      <header className="z-10 flex shrink-0 items-center gap-3 border-b border-slate-700 bg-[#182321] px-5 py-3.5 text-white shadow-lg sm:px-7">
        <div className="flex size-10 items-center justify-center rounded-lg bg-teal-500 text-xl text-[#10201d] shadow-[0_0_20px_rgba(45,212,191,.25)]">
          <Icon name="shield" />
        </div>
        <div className="min-w-0">
          <h1 className="text-base font-semibold tracking-tight sm:text-lg">IT Helpdesk</h1>
          <p className="text-[11px] font-medium uppercase tracking-[.11em] text-slate-300">AI Support Assistant</p>
        </div>
        <div className="ml-auto flex items-center gap-2 text-xs font-medium text-slate-200">
          <i className={`size-2 rounded-full ${connected ? "bg-emerald-400 shadow-[0_0_0_4px_rgba(74,222,128,.15)]" : "bg-red-400"}`} />
          {connected ? "Online" : "Reconnecting"}
        </div>
        <button onClick={startNewSession} className="ml-2 flex items-center gap-1.5 rounded-lg border border-teal-400/50 px-3 py-2 text-xs font-semibold text-teal-200 transition hover:bg-teal-400/10">
          <Icon name="plus" /> New Session
        </button>
      </header>

      <div className={`shrink-0 border-b px-5 py-2 text-[11px] font-semibold uppercase tracking-[.1em] sm:px-7 ${stage === "escalated" ? "border-red-200 bg-red-50 text-red-800" : "border-slate-200 bg-white text-slate-500"}`}>
        {stage === "escalated" ? "Human IT Support Required" : `Current stage · ${labels[stage] || stage}`}
      </div>

      <div className="flex min-h-0 flex-1">
        <section className="flex min-w-0 flex-1 flex-col border-r border-slate-200 bg-[#f8faf9]">
          <div className="flex flex-1 flex-col gap-4 overflow-y-auto px-4 py-6 sm:px-7">
            {!messages.length && (
              <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center">
                <div className="mb-8 text-center">
                  <div className="mx-auto mb-4 flex size-14 items-center justify-center rounded-2xl bg-teal-100 text-2xl text-teal-800">
                    <Icon name="spark" />
                  </div>
                  <h2 className="text-2xl font-semibold tracking-tight text-slate-900">How can IT support help?</h2>
                  <p className="mt-2 text-sm text-slate-500">Choose a category or describe the issue below.</p>
                </div>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                  {categories.map((category) => (
                    <button
                      key={category}
                      onClick={() => setText(category)}
                      className="rounded-xl border border-slate-200 bg-white px-3 py-4 text-left text-sm font-medium text-slate-700 shadow-sm transition hover:border-teal-400 hover:bg-teal-50 hover:text-teal-900"
                    >
                      <span className="mb-2 flex size-7 items-center justify-center rounded-md bg-slate-100 text-xs text-teal-700">
                        <Icon name="device" />
                      </span>
                      {category}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex max-w-full gap-2.5 ${message.role === "user" ? "flex-row-reverse" : message.role === "system" ? "justify-center" : ""}`}
              >
                <div
                  className={`${message.role === "system" ? "hidden" : "mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg text-[11px] font-bold"} ${message.role === "ai" ? "bg-teal-600 text-white" : "bg-slate-700 text-white"}`}
                >
                  {message.role === "ai" ? <Icon name="spark" /> : <Icon name="user" />}
                </div>
                <div
                  className={`max-w-[min(78%,620px)] break-words rounded-2xl px-4 py-3 text-sm leading-[1.55] ${
                    message.role === "user"
                      ? "rounded-tr-md bg-[#263735] text-white"
                      : message.role === "system"
                      ? "max-w-none border border-slate-200 bg-white px-3 py-1.5 text-center text-xs text-slate-500"
                      : message.escalate
                      ? "rounded-tl-md border border-red-200 bg-red-50 text-red-950"
                      : "rounded-tl-md border border-slate-200 bg-white text-slate-700 shadow-sm"
                  }`}
                >
                  {message.escalate && (
                    <div className="mb-2 text-xs font-bold uppercase tracking-[.09em] text-red-700">Human IT Support Required</div>
                  )}
                  {message.text}
                  {message.role === "ai" && message.confidence !== undefined && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      <span className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-semibold ring-1 ring-inset ${confidenceTone(message.confidence)}`}>
                        Confidence {Math.round(message.confidence * 100)}%
                      </span>
                      {message.label && (
                        <span className="inline-block rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-semibold capitalize text-slate-600">
                          {message.label.replaceAll("_", " ")}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEnd} />
          </div>

          <form onSubmit={send} className="flex shrink-0 items-end gap-2 border-t border-slate-200 bg-white px-4 py-4 sm:px-7">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  (e.currentTarget.form as HTMLFormElement).requestSubmit();
                }
              }}
              placeholder="Describe your IT issue…"
              className="min-h-[44px] flex-1 resize-none rounded-lg border border-slate-300 bg-slate-50 px-3.5 py-2.5 text-sm leading-6 outline-none placeholder:text-slate-400 focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
            />
            <button
              disabled={!connected}
              aria-label="Send message"
              className="flex size-11 shrink-0 items-center justify-center rounded-lg bg-teal-600 text-lg font-bold text-white shadow-sm transition hover:bg-teal-700 disabled:opacity-45"
            >
              <Icon name="send" />
            </button>
          </form>
        </section>

        <aside className="w-[400px] shrink-0 overflow-y-auto bg-[#f4f7f6] px-5 py-5 max-[1050px]:w-[340px] max-[760px]:hidden">
          <div className="flex flex-col gap-3.5">
            <Card title="Current stage">
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm text-slate-500">Support workflow</span>
                <span className={`rounded-full px-2.5 py-1 text-[10px] font-bold uppercase tracking-[.08em] ring-1 ring-inset ${stageTone(stage)}`}>
                  {labels[stage] || stage}
                </span>
              </div>
              <div className="mt-3 border-t border-slate-100 pt-3 text-[11px] text-slate-400">
                <span>Session ID</span>
                <div className="mt-1 break-all font-mono text-slate-500">{session || "Connecting…"}</div>
              </div>
            </Card>

            <Card title="Chat history">
              <div className="max-h-48 space-y-1 overflow-y-auto">
                {history.length ? history.map((item) => (
                  <button
                    key={item.session_id}
                    onClick={() => void openSession(item.session_id)}
                    className={`w-full rounded-lg px-2.5 py-2 text-left transition hover:bg-teal-50 ${item.session_id === session ? "bg-teal-50" : ""}`}
                  >
                    <div className="truncate text-sm font-medium text-slate-700">{item.preview}</div>
                    <div className="mt-0.5 text-[10px] uppercase tracking-[.08em] text-slate-400">{labels[item.stage] || item.stage}</div>
                  </button>
                )) : <p className="text-sm italic text-slate-400">No saved sessions yet.</p>}
              </div>
            </Card>

            <Card title="AI classification">
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-500">Confidence</span>
                <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset ${confidenceTone(meta.confidence)}`}>
                  {meta.confidence === undefined ? "—" : `${Math.round(meta.confidence * 100)}%`}
                </span>
              </div>
              <div className="mt-3 flex items-center justify-between gap-3 border-t border-slate-100 pt-3 text-sm">
                <span className="text-slate-500">Routing label</span>
                <span className="max-w-[58%] rounded-full bg-slate-100 px-2.5 py-1 text-right text-xs font-medium capitalize text-slate-700">
                  {(meta.predicted_escalation_label || "normal").replaceAll("_", " ")}
                </span>
              </div>
            </Card>

            <Card title="Employee / issue / device information">
              <div className="max-h-40 overflow-y-auto">
                {Object.keys(qualification).length ? (
                  Object.entries(qualification).map(([key, value]) => (
                    <div key={key} className="border-b border-slate-100 py-2 last:border-0">
                      <div className="text-[10px] font-semibold uppercase tracking-[.08em] text-slate-400">{key.replaceAll("_", " ")}</div>
                      <div className="mt-0.5 text-sm text-slate-700">{value}</div>
                    </div>
                  ))
                ) : (
                  <p className="text-sm italic text-slate-400">No information collected yet.</p>
                )}
              </div>
            </Card>

            <Card title="SOP gaps">
              <div className="max-h-32 overflow-y-auto">
                {gaps.length ? (
                  gaps.map((gap, index) => (
                    <div key={index} className="border-b border-slate-100 py-2 text-sm leading-snug text-amber-800 last:border-0">
                      {gap}
                    </div>
                  ))
                ) : (
                  <p className="text-sm italic text-slate-400">No SOP gaps identified.</p>
                )}
              </div>
            </Card>

            {stage === "escalated" && (
              <Card title="Escalation reason" warning>
                <div className="flex gap-2">
                  <span className="mt-0.5 text-red-600">●</span>
                  <div>
                    <p className="text-sm font-semibold text-red-900">Human IT Support Required</p>
                    {escalationReason && <p className="mt-1 text-sm leading-snug text-red-800">{escalationReason}</p>}
                  </div>
                </div>
              </Card>
            )}

            <button onClick={startNewSession} className="w-full rounded-lg border border-teal-600 bg-white px-3 py-2.5 text-sm font-semibold text-teal-800 transition hover:bg-teal-50">
              Start New Session
            </button>
          </div>
        </aside>
      </div>
    </main>
  );
}
