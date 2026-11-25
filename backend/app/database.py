import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional


def parse_sqlite_path(database_url: str) -> str:
    """Convert a DATABASE_URL like sqlite:////data/app.db into a filesystem path."""
    url = database_url.strip()
    if url in {":memory:", "sqlite:///:memory:", "sqlite://:memory:"}:
        return ":memory:"
    if url.startswith("sqlite:////"):
        return "/" + url[len("sqlite:////"):]
    if url.startswith("sqlite:///"):
        return url[len("sqlite:///"):]
    if url.startswith("sqlite://"):
        return url[len("sqlite://"):]
    return url


class Database:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.db_path = parse_sqlite_path(database_url)
        self._memory_conn: Optional[sqlite3.Connection] = None

        if self.db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(self.db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        else:
            # Keep a single shared connection open for in-memory test databases
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row

        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self._memory_conn is not None:
            try:
                yield self._memory_conn
                self._memory_conn.commit()
            except Exception:
                self._memory_conn.rollback()
                raise
            return

        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages (session_id, id);
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    system_prompt TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def check_health(self) -> bool:
        try:
            with self.connect() as conn:
                row = conn.execute("SELECT 1 AS ok;").fetchone()
                return bool(row and row["ok"] == 1)
        except Exception:
            return False

    def get_session_system_prompt(self, session_id: str) -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?;",
                (session_id,),
            ).fetchone()
            if row and row["system_prompt"]:
                return str(row["system_prompt"])
            return None

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC;
                """,
                (session_id,),
            ).fetchall()

        result: List[Dict[str, Any]] = []
        for row in rows:
            item: Dict[str, Any] = {
                "role": row["role"],
                "content": row["content"],
            }
            if row["role"] == "assistant" and row["metadata"]:
                try:
                    item["metadata"] = json.loads(row["metadata"])
                except json.JSONDecodeError:
                    item["metadata"] = {}
            result.append(item)
        return result

    def save_turn(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        assistant_metadata: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> None:
        """Atomically persist both user and assistant messages after a successful reply."""
        now = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(assistant_metadata)

        with self.connect() as conn:
            existing = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?;",
                (session_id,),
            ).fetchone()

            if existing is None:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, system_prompt, created_at, updated_at)
                    VALUES (?, ?, ?, ?);
                    """,
                    (session_id, system_prompt, now, now),
                )
            elif system_prompt is not None:
                conn.execute(
                    """
                    UPDATE sessions
                    SET system_prompt = ?, updated_at = ?
                    WHERE session_id = ?;
                    """,
                    (system_prompt, now, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?;",
                    (now, session_id),
                )

            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES (?, 'user', ?, NULL, ?);
                """,
                (session_id, user_content, now),
            )
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES (?, 'assistant', ?, ?, ?);
                """,
                (session_id, assistant_content, metadata_json, now),
            )
