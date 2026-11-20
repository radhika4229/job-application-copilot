import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_db_path(custom_path: str | None = None) -> str:
    return custom_path or os.getenv("DB_PATH", "job_copilot.db")


def init_db(db_path: str | None = None) -> None:
    path = get_db_path(db_path)
    parent = Path(path).parent
    if str(parent) != "." and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                jd_text TEXT NOT NULL,
                result_json TEXT,
                status TEXT NOT NULL,
                error_message TEXT
            );
            """
        )
        conn.commit()
    logger.info("Database initialized at '%s'", path)


def save_run(
    run_id: str,
    jd_text: str,
    status: str,
    result_json: str | None = None,
    error_message: str | None = None,
    created_at: str | None = None,
    db_path: str | None = None
) -> dict[str, Any]:
    path = get_db_path(db_path)
    ts = created_at or datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO runs (id, created_at, jd_text, result_json, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (run_id, ts, jd_text, result_json, status, error_message)
        )
        conn.commit()

    logger.info("Saved run %s with status '%s' in %s", run_id, status, path)
    return {
        "id": run_id,
        "created_at": ts,
        "jd_text": jd_text,
        "result_json": result_json,
        "status": status,
        "error_message": error_message
    }


def get_run(run_id: str, db_path: str | None = None) -> dict[str, Any] | None:
    path = get_db_path(db_path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, created_at, jd_text, result_json, status, error_message
            FROM runs
            WHERE id = ?;
            """,
            (run_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        data = dict(row)
        if data["result_json"]:
            try:
                data["result"] = json.loads(data["result_json"])
            except json.JSONDecodeError:
                data["result"] = None
        else:
            data["result"] = None

        return data


def list_runs(limit: int = 50, db_path: str | None = None) -> list[dict[str, Any]]:
    path = get_db_path(db_path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, created_at, status
            FROM runs
            ORDER BY datetime(created_at) DESC, rowid DESC
            LIMIT ?;
            """,
            (limit,)
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
