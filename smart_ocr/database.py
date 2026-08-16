"""SQLite persistence layer for OCR records (Database Collection of the synopsis)."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS ocr_records (
    ocr_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT    NOT NULL DEFAULT 'guest',
    image_name       TEXT    NOT NULL,
    extracted_text   TEXT    NOT NULL,
    confidence_score REAL    NOT NULL,
    created_at       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ocr_records_user ON ocr_records(user_id);
"""


class Database:
    """Thin wrapper around sqlite3 so every module talks to one storage API."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def save_record(
        self,
        image_name: str,
        extracted_text: str,
        confidence_score: float,
        user_id: str = "guest",
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO ocr_records (user_id, image_name, extracted_text, "
                "confidence_score, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    user_id,
                    image_name,
                    extracted_text,
                    round(float(confidence_score), 2),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            return int(cur.lastrowid)

    def get_record(self, ocr_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ocr_records WHERE ocr_id = ?", (ocr_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_records(self, user_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query = "SELECT * FROM ocr_records"
        params: list[Any] = []
        if user_id:
            query += " WHERE user_id = ?"
            params.append(user_id)
        query += " ORDER BY ocr_id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def delete_record(self, ocr_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM ocr_records WHERE ocr_id = ?", (ocr_id,))
            return cur.rowcount > 0
