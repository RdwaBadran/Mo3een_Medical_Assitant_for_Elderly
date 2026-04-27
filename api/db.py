"""
api/db.py
---------
SQLite persistence layer for multi-session chat history.

Stores sessions in `data/sessions.db` alongside the existing data directory.
Each session holds a JSON blob of messages (raw dicts, never LangChain objects).
All queries filter by BOTH thread_id AND user_id for data isolation.
"""

import sqlite3
import json
import os
import uuid
from datetime import datetime, timezone

# ── Database path ────────────────────────────────────────────────────────────
_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_DB_PATH = os.path.join(_DB_DIR, "sessions.db")


def _get_connection() -> sqlite3.Connection:
    """
    Returns a new SQLite connection with row_factory set to sqlite3.Row
    so results can be accessed by column name.
    """
    os.makedirs(_DB_DIR, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ══════════════════════════════════════════════════════════════════════════════
# Schema setup
# ══════════════════════════════════════════════════════════════════════════════

def create_table_if_not_exists() -> None:
    """
    Creates the `sessions` table and index if they don't already exist.
    Safe to call multiple times (idempotent).
    """
    conn = _get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id    TEXT PRIMARY KEY,
                session_name TEXT,
                user_id      TEXT NOT NULL,
                created_at   TEXT NOT NULL,
                messages     TEXT NOT NULL DEFAULT '[]'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user
            ON sessions(user_id)
        """)
        conn.commit()
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# CRUD helpers
# ══════════════════════════════════════════════════════════════════════════════

def create_session(user_id: str) -> str:
    """
    Create a new empty session for the given user.

    Args:
        user_id: The user identifier.

    Returns:
        thread_id: A new UUID string identifying this session.
    """
    thread_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO sessions (thread_id, session_name, user_id, created_at, messages) "
            "VALUES (?, ?, ?, ?, ?)",
            (thread_id, None, user_id, created_at, "[]"),
        )
        conn.commit()
    finally:
        conn.close()

    return thread_id


def list_sessions(user_id: str) -> list[dict]:
    """
    List all sessions for a user, ordered newest first.

    Args:
        user_id: The user identifier.

    Returns:
        List of dicts with keys: thread_id, session_name, created_at.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT thread_id, session_name, created_at "
            "FROM sessions WHERE user_id = ? "
            "ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [
            {
                "thread_id": row["thread_id"],
                "session_name": row["session_name"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_session(thread_id: str, user_id: str) -> dict | None:
    """
    Fetch a single session by thread_id, verifying user_id ownership.

    Args:
        thread_id: The session's UUID.
        user_id:   The user identifier (security check).

    Returns:
        Dict with thread_id, session_name, created_at, messages (parsed list),
        or None if not found / user mismatch.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT thread_id, session_name, user_id, created_at, messages "
            "FROM sessions WHERE thread_id = ? AND user_id = ?",
            (thread_id, user_id),
        ).fetchone()

        if row is None:
            return None

        return {
            "thread_id": row["thread_id"],
            "session_name": row["session_name"],
            "created_at": row["created_at"],
            "messages": json.loads(row["messages"]),
        }
    finally:
        conn.close()


def update_session_messages(
    thread_id: str,
    user_id: str,
    messages: list[dict],
    session_name: str | None = None,
) -> None:
    """
    Update the messages JSON blob for a session.
    Optionally updates the session_name (e.g. auto-generated from first message).

    Args:
        thread_id:    The session's UUID.
        user_id:      The user identifier (security check).
        messages:     List of raw message dicts: [{"role": "user"/"assistant", "content": "..."}]
        session_name: If provided, updates the session name.
    """
    conn = _get_connection()
    try:
        if session_name is not None:
            conn.execute(
                "UPDATE sessions SET messages = ?, session_name = ? "
                "WHERE thread_id = ? AND user_id = ?",
                (json.dumps(messages, ensure_ascii=False), session_name, thread_id, user_id),
            )
        else:
            conn.execute(
                "UPDATE sessions SET messages = ? "
                "WHERE thread_id = ? AND user_id = ?",
                (json.dumps(messages, ensure_ascii=False), thread_id, user_id),
            )
        conn.commit()
    finally:
        conn.close()


def delete_session(thread_id: str, user_id: str) -> bool:
    """
    Deletes a session from the database.

    Args:
        thread_id: The session's UUID.
        user_id: The user identifier (security check).

    Returns:
        bool: True if a row was deleted, False if not found.
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM sessions WHERE thread_id = ? AND user_id = ?",
            (thread_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def rename_session(thread_id: str, user_id: str, new_name: str) -> bool:
    """
    Renames a session.

    Args:
        thread_id: The session's UUID.
        user_id: The user identifier (security check).
        new_name: The new name for the session.

    Returns:
        bool: True if a row was updated, False if not found.
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "UPDATE sessions SET session_name = ? WHERE thread_id = ? AND user_id = ?",
            (new_name, thread_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
