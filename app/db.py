"""SQLite persistence: scheduled tasks, conversation history, audit log.

Design rule: scheduled tasks must be durable. A restart, a power cut, or an
expired token must never silently lose a pending job.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from . import build

ROOT = Path(__file__).resolve().parent.parent
# A hosted copy keeps its data on a mounted volume, not inside the source tree.
DATA_DIR = Path(os.environ.get("DATA_DIR") or (ROOT / "data"))
DB_PATH = DATA_DIR / "app.db"

_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    device_id    TEXT,
    device_name  TEXT,
    actions      TEXT NOT NULL,
    rule         TEXT NOT NULL,
    next_run_at  TEXT,
    enabled      INTEGER DEFAULT 1,
    created_at   TEXT,
    last_run_at  TEXT,
    last_result  TEXT,
    repeat_label TEXT,
    source_text  TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role       TEXT,
    content    TEXT,
    payload    TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    level      TEXT,
    event      TEXT,
    detail     TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_next ON jobs(enabled, next_run_at);
"""


@contextmanager
def conn():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=15)
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA journal_mode=WAL")
        yield c
        c.commit()
    finally:
        c.close()


def init() -> None:
    with _lock, conn() as c:
        c.executescript(SCHEMA)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------- jobs ----------------

def add_job(device_id: str, device_name: str, actions: list[dict],
            rule: dict, next_run_at: str, repeat_label: str = "",
            source_text: str = "") -> str:
    job_id = uuid.uuid4().hex[:10]
    with _lock, conn() as c:
        c.execute(
            "INSERT INTO jobs (id, device_id, device_name, actions, rule, next_run_at,"
            " enabled, created_at, repeat_label, source_text)"
            " VALUES (?,?,?,?,?,?,1,?,?,?)",
            (job_id, device_id, device_name, json.dumps(actions, ensure_ascii=False),
             json.dumps(rule, ensure_ascii=False), next_run_at, now_iso(),
             repeat_label, source_text),
        )
    return job_id


def list_jobs(only_enabled: bool = False) -> list[dict]:
    sql = "SELECT * FROM jobs"
    if only_enabled:
        sql += " WHERE enabled=1"
    sql += " ORDER BY enabled DESC, next_run_at ASC"
    with _lock, conn() as c:
        rows = c.execute(sql).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["actions"] = json.loads(d["actions"])
        d["rule"] = json.loads(d["rule"])
        out.append(d)
    return out


def get_job(job_id: str) -> dict | None:
    with _lock, conn() as c:
        r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["actions"] = json.loads(d["actions"])
    d["rule"] = json.loads(d["rule"])
    return d


def due_jobs(now_iso_str: str) -> list[dict]:
    with _lock, conn() as c:
        rows = c.execute(
            "SELECT * FROM jobs WHERE enabled=1 AND next_run_at IS NOT NULL"
            " AND next_run_at <= ? ORDER BY next_run_at ASC",
            (now_iso_str,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["actions"] = json.loads(d["actions"])
        d["rule"] = json.loads(d["rule"])
        out.append(d)
    return out


def update_job_after_run(job_id: str, next_run_at: str | None,
                         result: str, disable: bool = False) -> None:
    with _lock, conn() as c:
        c.execute(
            "UPDATE jobs SET last_run_at=?, last_result=?, next_run_at=?, enabled=?"
            " WHERE id=?",
            (now_iso(), result, next_run_at, 0 if disable else 1, job_id),
        )


def delete_job(job_id: str) -> bool:
    with _lock, conn() as c:
        cur = c.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        return cur.rowcount > 0


def clear_jobs() -> int:
    with _lock, conn() as c:
        cur = c.execute("DELETE FROM jobs")
        return cur.rowcount


# ---------------- messages ----------------

def add_message(role: str, content: str, payload: dict | None = None) -> None:
    payload = dict(payload or {})
    # Which build wrote this row. A reply is a snapshot of the code that ran,
    # and the message outlives it: three times today an old row was read as the
    # answer to a new question, because nothing in the interface could say it
    # had been written by a build that no longer runs. The stamp travels with
    # the payload rather than the content, so it is data about the message.
    payload["build"] = build.STAMP
    with _lock, conn() as c:
        c.execute(
            "INSERT INTO messages (role, content, payload, created_at) VALUES (?,?,?,?)",
            (role, content, json.dumps(payload, ensure_ascii=False), now_iso()),
        )


def recent_messages(limit: int = 60) -> list[dict]:
    with _lock, conn() as c:
        rows = c.execute(
            "SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in reversed(rows):
        d = dict(r)
        d["payload"] = json.loads(d["payload"] or "{}")
        out.append(d)
    return out


def clear_messages() -> None:
    with _lock, conn() as c:
        c.execute("DELETE FROM messages")


# ---------------- logs ----------------

def log(event: str, detail: str = "", level: str = "info") -> None:
    with _lock, conn() as c:
        c.execute(
            "INSERT INTO logs (level, event, detail, created_at) VALUES (?,?,?,?)",
            (level, event, detail, now_iso()),
        )


def recent_logs(limit: int = 50) -> list[dict]:
    with _lock, conn() as c:
        rows = c.execute(
            "SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
