"""PostgreSQL persistence for Closira sessions and conversation messages."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS employee_sessions (
    session_id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stage TEXT NOT NULL,
    unanswered_count INTEGER NOT NULL DEFAULT 0,
    qualification JSONB NOT NULL DEFAULT '{}'::jsonb,
    sop_gaps JSONB NOT NULL DEFAULT '[]'::jsonb,
    conversation_ended BOOLEAN NOT NULL DEFAULT FALSE,
    intent_label TEXT NOT NULL DEFAULT 'normal',
    intent_confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    escalation_status BOOLEAN NOT NULL DEFAULT FALSE,
    escalation_reason TEXT,
    escalation_trigger TEXT,
    final_summary JSONB,
    summary_created_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES employee_sessions(session_id) ON DELETE CASCADE,
    message_index INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, message_index)
);

CREATE INDEX IF NOT EXISTS conversation_messages_session_id_idx
    ON conversation_messages (session_id, message_index);
"""


def database_url() -> str:
    """Return the required PostgreSQL connection string without logging credentials."""
    value = os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError("DATABASE_URL must be configured for PostgreSQL session persistence")
    return value


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    with psycopg.connect(database_url(), row_factory=dict_row) as conn:
        yield conn


def initialize_database() -> None:
    with connection() as conn:
        conn.execute(SCHEMA_SQL)


def _message_record(message: Any, index: int) -> tuple[int, str, str, dict[str, Any]] | None:
    role = message.type if hasattr(message, "type") else message.get("role", "user")
    role = "assistant" if role in {"ai", "assistant"} else "user" if role in {"human", "user"} else None
    if role is None:
        return None
    content = message.content if hasattr(message, "content") else message.get("content", "")
    metadata: dict[str, Any] = {}
    if role == "assistant":
        try:
            payload = json.loads(content)
            if isinstance(payload, dict):
                metadata = {key: value for key, value in payload.items() if key != "response"}
                content = str(payload.get("response", ""))
        except (TypeError, json.JSONDecodeError):
            pass
    return index, role, str(content), metadata


def create_session(state: dict[str, Any]) -> None:
    """Create the session row before the WebSocket accepts its first message."""
    persist_session(state)

def persist_session(state: dict[str, Any]) -> None:
    """Upsert the agent state and append only new conversation messages."""
    session_id = state["session_id"]
    escalation_status = (
        state.get("stage") == "escalated"
        or bool(state.get("escalation_reason"))
    )

    with connection() as conn:
        conn.execute(
            """
            INSERT INTO employee_sessions (
                session_id, stage, unanswered_count, qualification, sop_gaps,
                conversation_ended, intent_label, intent_confidence,
                escalation_status, escalation_reason, escalation_trigger
            ) VALUES (
                %(session_id)s, %(stage)s, %(unanswered_count)s, %(qualification)s::jsonb,
                %(sop_gaps)s::jsonb, %(conversation_ended)s, %(intent_label)s,
                %(intent_confidence)s, %(escalation_status)s, %(escalation_reason)s,
                %(escalation_trigger)s
            )
            ON CONFLICT (session_id) DO UPDATE SET
                updated_at = NOW(),
                stage = EXCLUDED.stage,
                unanswered_count = EXCLUDED.unanswered_count,
                qualification = EXCLUDED.qualification,
                sop_gaps = EXCLUDED.sop_gaps,
                conversation_ended = EXCLUDED.conversation_ended,
                intent_label = EXCLUDED.intent_label,
                intent_confidence = EXCLUDED.intent_confidence,
                escalation_status = EXCLUDED.escalation_status,
                escalation_reason = EXCLUDED.escalation_reason,
                escalation_trigger = EXCLUDED.escalation_trigger
            """,
            {
                "session_id": session_id,
                "stage": state["stage"],
                "unanswered_count": state["unanswered_count"],
                "qualification": json.dumps(state["qualification"]),
                "sop_gaps": json.dumps(state["sop_gaps"]),
                "conversation_ended": state["conversation_ended"],
                "intent_label": state["intent_label"],
                "intent_confidence": state["intent_confidence"],
                "escalation_status": escalation_status,
                "escalation_reason": state.get("escalation_reason") or None,
                "escalation_trigger": state.get("escalation_trigger") or None,
            },
        )

        records = [
            _message_record(message, index)
            for index, message in enumerate(state["messages"])
        ]

        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO conversation_messages
                    (session_id, message_index, role, content, metadata)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (session_id, message_index) DO NOTHING
                """,
                [
                    (
                        session_id,
                        index,
                        role,
                        content,
                        json.dumps(metadata),
                    )
                    for record in records
                    if record
                    for index, role, content, metadata in [record]
                ],
            )
def record_escalation(session_id: str, reason: str, trigger: str, intent_label: str, intent_confidence: float) -> None:
    with connection() as conn:
        conn.execute(
            """UPDATE employee_sessions SET updated_at = NOW(), stage = 'escalated', escalation_status = TRUE,
               escalation_reason = %(reason)s, escalation_trigger = %(trigger)s,
               intent_label = %(intent_label)s, intent_confidence = %(intent_confidence)s
               WHERE session_id = %(session_id)s""",
            {"session_id": session_id, "reason": reason, "trigger": trigger,
             "intent_label": intent_label, "intent_confidence": intent_confidence},
        )


def save_final_summary(session_id: str, summary: dict[str, Any]) -> None:
    with connection() as conn:
        conn.execute(
            """UPDATE employee_sessions SET updated_at = NOW(), final_summary = %(summary)s::jsonb,
               summary_created_at = %(created_at)s WHERE session_id = %(session_id)s""",
            {"session_id": session_id, "summary": json.dumps(summary),
             "created_at": datetime.now(timezone.utc)},
        )


def get_session(session_id: str) -> dict[str, Any] | None:
    """Return a session and its messages for persistence checks or internal use."""
    with connection() as conn:
        session = conn.execute("SELECT * FROM employee_sessions WHERE session_id = %s", (session_id,)).fetchone()
        if session is None:
            return None
        messages = conn.execute(
            "SELECT message_index, role, content, metadata, created_at FROM conversation_messages "
            "WHERE session_id = %s ORDER BY message_index", (session_id,)
        ).fetchall()
    session["messages"] = messages
    return session


def list_sessions() -> list[dict[str, Any]]:
    """Return session summaries for the chat-history view."""
    with connection() as conn:
        return conn.execute(
            """
            SELECT session_id, created_at, updated_at, stage,
                   COALESCE((
                       SELECT content FROM conversation_messages
                       WHERE session_id = employee_sessions.session_id AND role = 'user'
                       ORDER BY message_index LIMIT 1
                   ), 'New conversation') AS preview
            FROM employee_sessions
            ORDER BY updated_at DESC
            """
        ).fetchall()
