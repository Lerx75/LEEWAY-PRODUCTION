import json
import os
import sqlite3
import zlib
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet


class ProjectStore:
    def __init__(self, db_path: str, encryption_key: str) -> None:
        if not db_path:
            raise ValueError("db_path is required")
        if not encryption_key:
            raise ValueError("encryption_key is required")
        self._db_path = db_path
        try:
            self._fernet = Fernet(encryption_key)
        except Exception as exc:  # pragma: no cover - configuration error
            raise ValueError("PROJECT_ENCRYPTION_KEY must be a valid Fernet key") from exc
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = Lock()

    # ----------------------------
    # Internals
    # ----------------------------
    def _conn_guard(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    meta_json TEXT NOT NULL,
                    rows_blob BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id);")
            self._conn = conn
        return self._conn

    def _encode_rows(self, rows: List[List[Any]]) -> bytes:
        data = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        compressed = zlib.compress(data, level=6)
        return self._fernet.encrypt(compressed)

    def _decode_rows(self, blob: bytes) -> List[List[Any]]:
        decrypted = self._fernet.decrypt(blob)
        data = zlib.decompress(decrypted)
        return json.loads(data.decode("utf-8"))

    # ----------------------------
    # Public API
    # ----------------------------
    def save_project(
        self,
        *,
        user_id: str,
        name: str,
        mode: str,
        rows: List[List[Any]],
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        meta_json = json.dumps(meta or {}, ensure_ascii=False, separators=(",", ":"))
        payload = self._encode_rows(rows)
        with self._lock:
            conn = self._conn_guard()
            cur = conn.execute(
                """
                INSERT INTO projects (user_id, name, mode, meta_json, rows_blob, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, name, mode, meta_json, payload, now, now),
            )
            project_id = int(cur.lastrowid)
        return project_id

    def update_project_rows(
        self,
        *,
        project_id: int,
        user_id: str,
        rows: List[List[Any]],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        meta_json = json.dumps(meta or {}, ensure_ascii=False, separators=(",", ":"))
        payload = self._encode_rows(rows)
        with self._lock:
            conn = self._conn_guard()
            conn.execute(
                """
                UPDATE projects SET rows_blob=?, meta_json=?, updated_at=?
                WHERE id=? AND user_id=?
                """,
                (payload, meta_json, now, project_id, user_id),
            )

    def list_projects(self, *, user_id: str) -> List[Dict[str, Any]]:
        conn = self._conn_guard()
        res = conn.execute(
            """
            SELECT id, name, mode, meta_json, created_at, updated_at
            FROM projects
            WHERE user_id=?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()
        items: List[Dict[str, Any]] = []
        for row in res:
            meta = json.loads(row["meta_json"] or "{}")
            items.append(
                {
                    "id": int(row["id"]),
                    "name": row["name"],
                    "mode": row["mode"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "meta": meta,
                }
            )
        return items

    def get_project(self, *, project_id: int, user_id: str) -> Optional[Dict[str, Any]]:
        conn = self._conn_guard()
        row = conn.execute(
            """
            SELECT id, name, mode, meta_json, rows_blob, created_at, updated_at
            FROM projects
            WHERE id=? AND user_id=?
            """,
            (project_id, user_id),
        ).fetchone()
        if not row:
            return None
        rows = self._decode_rows(row["rows_blob"])
        meta = json.loads(row["meta_json"] or "{}")
        return {
            "id": int(row["id"]),
            "name": row["name"],
            "mode": row["mode"],
            "rows": rows,
            "meta": meta,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def delete_project(self, *, project_id: int, user_id: str) -> bool:
        with self._lock:
            conn = self._conn_guard()
            cur = conn.execute(
                "DELETE FROM projects WHERE id=? AND user_id=?",
                (project_id, user_id),
            )
            return cur.rowcount > 0


def build_store_from_env() -> Optional[ProjectStore]:
    db_path = os.getenv("PROJECT_DB_PATH", "projects.db").strip()
    key = os.getenv("PROJECT_ENCRYPTION_KEY", "").strip()
    if not key:
        return None
    return ProjectStore(db_path=db_path, encryption_key=key)
