from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from auth_utils import generate_token


class DBStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                  id TEXT PRIMARY KEY,
                  email TEXT UNIQUE NOT NULL,
                  name TEXT NOT NULL,
                  password_salt TEXT NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_tokens (
                  token TEXT PRIMARY KEY,
                  user_id TEXT NOT NULL,
                  expires_at INTEGER NOT NULL,
                  created_at INTEGER NOT NULL,
                  FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                  id TEXT PRIMARY KEY,
                  user_id TEXT NOT NULL,
                  state_json TEXT NOT NULL,
                  created_at INTEGER NOT NULL,
                  updated_at INTEGER NOT NULL,
                  FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  content TEXT NOT NULL,
                  created_at INTEGER NOT NULL,
                  FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_auth_tokens_user_id ON auth_tokens(user_id);
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
                """
            )

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
        return dict(row) if row else None

    def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def create_user(
        self,
        email: str,
        name: str,
        password_salt: str,
        password_hash: str,
    ) -> dict[str, Any]:
        now = int(time.time())
        user_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users (id, email, name, password_salt, password_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, email.lower(), name, password_salt, password_hash, now),
            )
            conn.commit()
        return {
            "id": user_id,
            "email": email.lower(),
            "name": name,
            "created_at": now,
        }

    def create_token(self, user_id: str, ttl_seconds: int = 7 * 24 * 3600) -> str:
        token = generate_token()
        now = int(time.time())
        expires_at = now + ttl_seconds
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO auth_tokens (token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
                (token, user_id, expires_at, now),
            )
            conn.commit()
        return token

    def get_user_by_token(self, token: str) -> dict[str, Any] | None:
        now = int(time.time())
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.email, u.name, u.created_at
                FROM auth_tokens t
                JOIN users u ON u.id = t.user_id
                WHERE t.token = ? AND t.expires_at > ?
                """,
                (token, now),
            ).fetchone()
        return dict(row) if row else None

    def revoke_token(self, token: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))
            conn.commit()

    def create_chat_session(self, user_id: str, state_json: str) -> str:
        now = int(time.time())
        session_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (id, user_id, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user_id, state_json, now, now),
            )
            conn.commit()
        return session_id

    def get_chat_session_state(self, session_id: str, user_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
        if not row:
            return None
        try:
            parsed = json.loads(str(row["state_json"]))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def get_latest_chat_session(self, user_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, state_json, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        if not row:
            return None
        try:
            state = json.loads(str(row["state_json"]))
            if not isinstance(state, dict):
                state = {}
        except Exception:
            state = {}
        return {
            "id": str(row["id"]),
            "state": state,
            "created_at": int(row["created_at"]),
            "updated_at": int(row["updated_at"]),
        }

    def list_chat_sessions(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 100))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  s.id,
                  s.state_json,
                  s.created_at,
                  s.updated_at,
                  (
                    SELECT m.content
                    FROM chat_messages m
                    WHERE m.session_id = s.id AND m.role = 'assistant'
                    ORDER BY m.id DESC
                    LIMIT 1
                  ) AS last_assistant_message
                FROM chat_sessions s
                WHERE s.user_id = ?
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (user_id, safe_limit),
            ).fetchall()

        sessions: list[dict[str, Any]] = []
        for row in rows:
            try:
                state = json.loads(str(row["state_json"]))
                if not isinstance(state, dict):
                    state = {}
            except Exception:
                state = {}

            sessions.append(
                {
                    "id": str(row["id"]),
                    "state": state,
                    "created_at": int(row["created_at"]),
                    "updated_at": int(row["updated_at"]),
                    "last_assistant_message": str(row["last_assistant_message"] or ""),
                }
            )
        return sessions

    def update_chat_session_state(self, session_id: str, user_id: str, state_json: str) -> bool:
        now = int(time.time())
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE chat_sessions
                SET state_json = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (state_json, now, session_id, user_id),
            )
            conn.commit()
        return cursor.rowcount > 0

    def append_chat_messages(self, session_id: str, messages: list[dict[str, str]]) -> None:
        if not messages:
            return
        now = int(time.time())
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                [
                    (
                        session_id,
                        str(item.get("role", "")),
                        str(item.get("content", "")),
                        now,
                    )
                    for item in messages
                ],
            )
            conn.commit()
