"""
Database layer — all SQLite operations in one place.

Public API: the `Database` class.  No raw SQL leaks outside this module.

Schema notes:
  - ticks are the fundamental unit; session_id is nullable (Executive
    actions are not inside sessions)
  - FTS5 virtual table `knowledge_fts` enables fast keyword search over
    knowledge file contents indexed here
  - journal_index / reflection_index / weekly_index provide lightweight
    metadata for content files that live on disk as markdown
  - We deliberately avoid storing large blobs; transcripts live on disk
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from .models import (
    Goal,
    GoalStatusChange,
    SessionResult,
    Task,
    TickRecord,
)

log = logging.getLogger(__name__)


# ── Schema ─────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS goals (
    goal_id         TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',
    priority        INTEGER NOT NULL DEFAULT 10,
    created_at_tick INTEGER NOT NULL,
    last_worked_tick INTEGER NOT NULL DEFAULT 0,
    total_ticks     INTEGER NOT NULL DEFAULT 0,
    workspace_path  TEXT NOT NULL,
    blocked_reason  TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_goals_status   ON goals(status);
CREATE INDEX IF NOT EXISTS idx_goals_priority ON goals(priority);

CREATE TABLE IF NOT EXISTS sessions (
    session_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id         TEXT NOT NULL,
    task_description TEXT NOT NULL,
    task_criteria   TEXT NOT NULL,
    status          TEXT,           -- filled in on completion
    summary         TEXT,
    tick_start      INTEGER NOT NULL,
    tick_end        INTEGER,
    action_count    INTEGER,
    tokens_used     INTEGER,
    transcript_path TEXT,
    FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_goal ON sessions(goal_id);

CREATE TABLE IF NOT EXISTS ticks (
    tick_id     INTEGER PRIMARY KEY,
    session_id  INTEGER,            -- NULL for executive/infra ticks
    goal_id     TEXT,
    actor       TEXT NOT NULL,
    action_type TEXT NOT NULL,
    summary     TEXT NOT NULL,
    outcome     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ticks_goal       ON ticks(goal_id);
CREATE INDEX IF NOT EXISTS idx_ticks_session    ON ticks(session_id);
CREATE INDEX IF NOT EXISTS idx_ticks_actor      ON ticks(actor);
CREATE INDEX IF NOT EXISTS idx_ticks_desc       ON ticks(tick_id DESC);

-- Lightweight knowledge index — path + one-line summary for fast listing
CREATE TABLE IF NOT EXISTS knowledge_index (
    topic       TEXT PRIMARY KEY,   -- e.g. "nginx"
    file_path   TEXT NOT NULL,      -- relative path inside workspace
    summary     TEXT NOT NULL DEFAULT '',
    updated_at_tick INTEGER NOT NULL DEFAULT 0
);

-- FTS5 over knowledge content for keyword search
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    topic,
    content,
    content='knowledge_index',
    content_rowid='rowid'
);

-- Lightweight journal index — metadata only; content lives on disk as markdown
CREATE TABLE IF NOT EXISTS journal_index (
    goal_id     TEXT NOT NULL,
    tick_start  INTEGER NOT NULL,
    tick_end    INTEGER NOT NULL,
    file_path   TEXT NOT NULL,      -- relative path inside workspace
    summary     TEXT NOT NULL DEFAULT '',  -- first line of the entry
    PRIMARY KEY (goal_id, tick_start),
    FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
);

CREATE INDEX IF NOT EXISTS idx_journal_goal ON journal_index(goal_id, tick_start DESC);

-- Lightweight reflection index — metadata only; content lives in meta/reflections/
CREATE TABLE IF NOT EXISTS reflection_index (
    tick            INTEGER PRIMARY KEY,
    trigger_reason  TEXT NOT NULL DEFAULT '',
    file_path       TEXT NOT NULL,
    summary         TEXT NOT NULL DEFAULT ''   -- first non-empty line of observations
);

-- Lightweight weekly-summary index — metadata only; content lives in meta/summaries/
CREATE TABLE IF NOT EXISTS weekly_index (
    tick        INTEGER PRIMARY KEY,
    file_path   TEXT NOT NULL,
    summary     TEXT NOT NULL DEFAULT ''   -- first line of the weekly summary
);
"""


class Database:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._conn: sqlite3.Connection | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def ensure_schema(self) -> None:
        with self._cursor() as cur:
            cur.executescript(_DDL)
        log.debug("Schema OK: %s", self._path)

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        assert self._conn is not None, "Database.connect() was not called"
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # ── Goals ──────────────────────────────────────────────────────────────

    def create_goal(self, goal: Goal) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO goals
                  (goal_id, title, description, status, priority,
                   created_at_tick, last_worked_tick, total_ticks,
                   workspace_path, blocked_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    goal.goal_id, goal.title, goal.description,
                    goal.status, goal.priority, goal.created_at_tick,
                    goal.last_worked_tick, goal.total_ticks,
                    goal.workspace_path, goal.blocked_reason,
                ),
            )

    def get_goal(self, goal_id: str) -> Goal | None:
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT * FROM goals WHERE goal_id = ?", (goal_id,)
            ).fetchone()
        return _row_to_goal(row) if row else None

    def get_active_goals(self) -> list[Goal]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM goals WHERE status = 'active' ORDER BY priority ASC"
            ).fetchall()
        return [_row_to_goal(r) for r in rows]

    def get_all_goals(self) -> list[Goal]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM goals ORDER BY priority ASC"
            ).fetchall()
        return [_row_to_goal(r) for r in rows]

    def update_goal(self, goal_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._cursor() as cur:
            cur.execute(
                f"UPDATE goals SET {cols} WHERE goal_id = ?",
                (*fields.values(), goal_id),
            )

    def get_goal_counts_by_status(self) -> dict[str, int]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT status, COUNT(*) as cnt FROM goals GROUP BY status"
            ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    def get_neglected_goals(self, threshold_tick: int) -> list[Goal]:
        """Active goals not worked on since before threshold_tick."""
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT * FROM goals
                WHERE status = 'active'
                  AND last_worked_tick < ?
                ORDER BY last_worked_tick ASC
                """,
                (threshold_tick,),
            ).fetchall()
        return [_row_to_goal(r) for r in rows]

    # ── Sessions ───────────────────────────────────────────────────────────

    def open_session(
        self,
        goal_id: str,
        task: Task,
        tick_start: int,
        transcript_path: str,
    ) -> int:
        """Insert a new session row (status=NULL) and return its id."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions
                  (goal_id, task_description, task_criteria,
                   tick_start, transcript_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    goal_id, task.description, task.criteria,
                    tick_start, transcript_path,
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def complete_session(self, session_id: int, result: SessionResult) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                UPDATE sessions SET
                  status       = ?,
                  summary      = ?,
                  tick_end     = ?,
                  action_count = ?,
                  tokens_used  = ?
                WHERE session_id = ?
                """,
                (
                    result.status, result.summary, result.tick_end,
                    result.action_count, result.tokens_used,
                    session_id,
                ),
            )

    def get_recent_sessions(
        self, n: int, goal_id: str | None = None
    ) -> list[dict]:
        with self._cursor() as cur:
            if goal_id:
                rows = cur.execute(
                    """
                    SELECT session_id, goal_id, status, summary,
                           tick_start, tick_end, action_count
                    FROM sessions
                    WHERE goal_id = ?
                    ORDER BY session_id DESC LIMIT ?
                    """,
                    (goal_id, n),
                ).fetchall()
            else:
                rows = cur.execute(
                    """
                    SELECT session_id, goal_id, status, summary,
                           tick_start, tick_end, action_count
                    FROM sessions
                    ORDER BY session_id DESC LIMIT ?
                    """,
                    (n,),
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Ticks ──────────────────────────────────────────────────────────────

    def log_tick(self, tick: TickRecord) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO ticks
                  (tick_id, session_id, goal_id, actor, action_type, summary, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tick.tick_id, tick.session_id, tick.goal_id,
                    tick.actor, tick.action_type, tick.summary, tick.outcome,
                ),
            )

    def get_last_tick_id(self) -> int:
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT MAX(tick_id) as last_id FROM ticks"
            ).fetchone()
        return row["last_id"] or 0

    def get_recent_ticks(
        self, n: int, goal_id: str | None = None
    ) -> list[TickRecord]:
        with self._cursor() as cur:
            if goal_id:
                rows = cur.execute(
                    """
                    SELECT * FROM ticks WHERE goal_id = ?
                    ORDER BY tick_id DESC LIMIT ?
                    """,
                    (goal_id, n),
                ).fetchall()
            else:
                rows = cur.execute(
                    "SELECT * FROM ticks ORDER BY tick_id DESC LIMIT ?", (n,)
                ).fetchall()
        return [_row_to_tick(r) for r in reversed(rows)]

    def get_last_n_summaries(self, n: int) -> list[str]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT summary FROM ticks ORDER BY tick_id DESC LIMIT ?", (n,)
            ).fetchall()
        return [r["summary"] for r in rows]

    def get_last_n_outcomes(self, n: int) -> list[str]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT outcome FROM ticks ORDER BY tick_id DESC LIMIT ?", (n,)
            ).fetchall()
        return [r["outcome"] for r in rows]

    def get_ticks_range(self, start: int, end: int) -> list[TickRecord]:
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT * FROM ticks
                WHERE tick_id >= ? AND tick_id <= ?
                ORDER BY tick_id ASC
                """,
                (start, end),
            ).fetchall()
        return [_row_to_tick(r) for r in rows]

    def archive_ticks_before(self, tick_id: int, archive_path: str) -> int:
        """
        Export all ticks with tick_id < tick_id to a JSONL archive file,
        then delete them from the DB.  Returns the count of archived rows.
        """
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM ticks WHERE tick_id < ? ORDER BY tick_id", (tick_id,)
            ).fetchall()

        if not rows:
            return 0

        archive = Path(archive_path)
        archive.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("a") as f:
            for row in rows:
                f.write(json.dumps(dict(row)) + "\n")

        with self._cursor() as cur:
            cur.execute("DELETE FROM ticks WHERE tick_id < ?", (tick_id,))

        log.info("Archived %d ticks to %s", len(rows), archive_path)
        return len(rows)

    # ── Knowledge index ────────────────────────────────────────────────────

    def upsert_knowledge(
        self, topic: str, file_path: str, summary: str, tick: int
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO knowledge_index (topic, file_path, summary, updated_at_tick)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(topic) DO UPDATE SET
                  file_path      = excluded.file_path,
                  summary        = excluded.summary,
                  updated_at_tick = excluded.updated_at_tick
                """,
                (topic, file_path, summary, tick),
            )

    def search_knowledge(self, query: str, limit: int = 5) -> list[dict]:
        """FTS5 keyword search over knowledge content."""
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT k.topic, k.file_path, k.summary
                FROM knowledge_fts f
                JOIN knowledge_index k ON k.rowid = f.rowid
                WHERE knowledge_fts MATCH ?
                ORDER BY rank LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_knowledge(self) -> list[dict]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT topic, file_path, summary, updated_at_tick "
                "FROM knowledge_index ORDER BY topic"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Journal index ──────────────────────────────────────────────────────

    def upsert_journal(
        self,
        goal_id: str,
        tick_start: int,
        tick_end: int,
        file_path: str,
        summary: str = "",
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO journal_index
                  (goal_id, tick_start, tick_end, file_path, summary)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(goal_id, tick_start) DO UPDATE SET
                  tick_end  = excluded.tick_end,
                  file_path = excluded.file_path,
                  summary   = excluded.summary
                """,
                (goal_id, tick_start, tick_end, file_path, summary),
            )

    def get_recent_journals(
        self, goal_id: str, n: int
    ) -> list[dict]:
        """Return the n most-recent journal index entries for a goal."""
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT goal_id, tick_start, tick_end, file_path, summary
                FROM journal_index
                WHERE goal_id = ?
                ORDER BY tick_start DESC LIMIT ?
                """,
                (goal_id, n),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def list_journals_for_goal(self, goal_id: str) -> list[dict]:
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT goal_id, tick_start, tick_end, file_path, summary
                FROM journal_index WHERE goal_id = ?
                ORDER BY tick_start ASC
                """,
                (goal_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Reflection index ───────────────────────────────────────────────────

    def upsert_reflection(
        self,
        tick: int,
        trigger_reason: str,
        file_path: str,
        summary: str = "",
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO reflection_index
                  (tick, trigger_reason, file_path, summary)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tick) DO UPDATE SET
                  trigger_reason = excluded.trigger_reason,
                  file_path      = excluded.file_path,
                  summary        = excluded.summary
                """,
                (tick, trigger_reason, file_path, summary),
            )

    def get_recent_reflections(self, n: int) -> list[dict]:
        """Return the n most-recent reflection index entries."""
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT tick, trigger_reason, file_path, summary
                FROM reflection_index
                ORDER BY tick DESC LIMIT ?
                """,
                (n,),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ── Weekly summary index ───────────────────────────────────────────────

    def upsert_weekly(
        self, tick: int, file_path: str, summary: str = ""
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO weekly_index (tick, file_path, summary)
                VALUES (?, ?, ?)
                ON CONFLICT(tick) DO UPDATE SET
                  file_path = excluded.file_path,
                  summary   = excluded.summary
                """,
                (tick, file_path, summary),
            )

    def get_recent_weeklies(self, n: int) -> list[dict]:
        """Return the n most-recent weekly summary index entries."""
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT tick, file_path, summary
                FROM weekly_index
                ORDER BY tick DESC LIMIT ?
                """,
                (n,),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]


# ── Private helpers ────────────────────────────────────────────────────────

def _row_to_goal(row: sqlite3.Row) -> Goal:
    return Goal(
        goal_id=row["goal_id"],
        title=row["title"],
        description=row["description"],
        status=row["status"],
        priority=row["priority"],
        created_at_tick=row["created_at_tick"],
        last_worked_tick=row["last_worked_tick"],
        total_ticks=row["total_ticks"],
        workspace_path=row["workspace_path"],
        blocked_reason=row["blocked_reason"] or "",
    )


def _row_to_tick(row: sqlite3.Row) -> TickRecord:
    return TickRecord(
        tick_id=row["tick_id"],
        session_id=row["session_id"],
        goal_id=row["goal_id"],
        actor=row["actor"],
        action_type=row["action_type"],
        summary=row["summary"],
        outcome=row["outcome"],
    )
