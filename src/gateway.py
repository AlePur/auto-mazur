"""
GatewayServer — lightweight stdlib HTTP server that exposes the agent's
internal state for observation and allows the user to write inbox messages.

Backend-only.  No HTML is served.  All responses are JSON (Content-Type:
application/json).  The server is started as a daemon thread alongside the
agent loop and shares the same Database connection (WAL mode allows safe
concurrent reads while the agent writes).

Endpoints
─────────
Observe
  GET /status                             current tick + goal counts
  GET /goals                              all goals
  GET /goals/{id}                         single goal
  GET /goals/{id}/sessions?n=20           recent sessions for goal
  GET /goals/{id}/journals                journal index for goal
  GET /goals/{id}/journals/{tick_start}   journal markdown content
  GET /goals/{id}/checkpoint              CHECKPOINT.md content
  GET /sessions?n=20                      recent sessions (all goals)
  GET /ticks?n=50&goal_id=X              recent ticks
  GET /knowledge                          knowledge index
  GET /knowledge/{topic}                  knowledge markdown content
  GET /weeklies?n=5                       recent weekly-summary index entries
  GET /weeklies/{tick}                    weekly-summary markdown content
  GET /outbox?n=20                        recent outbox messages
  GET /inbox?n=20                         recent inbox messages (any state)
  GET /files/{rel_path}                   any file under workspace root
                                          (path traversal is blocked)
Audit
  GET /audit/llm?n=50&since=<ts>         recent LLM calls (ring buffer)
  GET /audit/tools?n=100&since=<ts>      recent tool executions (ring buffer)
  GET /audit/llm/dates                    available LLM audit dates
  GET /audit/tools/dates                  available tool audit dates
  GET /audit/llm/history?date=YYYY-MM-DD LLM JSONL for a specific date
  GET /audit/tools/history?date=YYYY-MM-DD tool JSONL for a specific date

Interact
  POST /inbox    body: {"text": "…"}      inject a user message
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from .audit import AuditLogger
    from .db import Database
    from .store import Store
    from .workspace import Workspace

log = logging.getLogger(__name__)

# Maximum size of a workspace file we'll serve inline (1 MB).
_MAX_FILE_BYTES = 1_048_576


class GatewayServer:
    """
    Thin wrapper around HTTPServer that keeps shared state accessible to
    request handlers via a thread-local-friendly class variable.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        db: "Database",
        store: "Store",
        audit: "AuditLogger | None",
        loop: Any,  # MainLoop — imported lazily to avoid circular dependency
        workspace: "Workspace | None" = None,
    ) -> None:
        self._host = host
        self._port = port

        # Inject shared state into the handler class.  Because BaseHTTP-
        # RequestHandler is instantiated fresh per request, we use class-level
        # attributes rather than instance attributes.
        _Handler.db = db
        _Handler.store = store
        _Handler.workspace = workspace
        _Handler.audit = audit
        _Handler.loop = loop

        self._server = HTTPServer((host, port), _Handler)
        self._server.timeout = 1.0  # allow clean shutdown checks

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()


# ── Request handler ────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    # Populated by GatewayServer.__init__ — shared across all request instances
    db: "Database"
    store: "Store"
    workspace: "Workspace | None" = None  # only for /files endpoint
    audit: "AuditLogger | None" = None
    loop: Any = None

    # ── Routing ────────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)

        try:
            data = self._route_get(path, qs)
            self._send_json(200, data)
        except _NotFound as exc:
            self._send_json(404, {"error": str(exc)})
        except _BadRequest as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            log.exception("Gateway GET error: %s", exc)
            self._send_json(500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(length) if length else b""
        try:
            body = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"Invalid JSON: {exc}"})
            return

        try:
            data = self._route_post(path, body)
            self._send_json(200, data)
        except _NotFound as exc:
            self._send_json(404, {"error": str(exc)})
        except _BadRequest as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            log.exception("Gateway POST error: %s", exc)
            self._send_json(500, {"error": str(exc)})

    # ── GET router ─────────────────────────────────────────────────────────

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Handle CORS preflight requests (sent before POST /inbox)."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _route_get(self, path: str, qs: dict) -> Any:  # noqa: C901
        db = self.db
        store = self.store

        # /status
        if path == "/status":
            # Response:
            #   {
            #     "tick":        int,             -- ID of the last completed tick (0 if none yet)
            #     "goal_counts": {<status>: int}  -- e.g. {"active": 2, "blocked": 1, "done": 5}
            #   }
            # goal_counts only includes statuses that have at least one goal;
            # possible status keys: "active", "blocked", "done", "abandoned", "paused".
            return {
                "tick": db.get_last_tick_id(),
                "goal_counts": db.get_goal_counts_by_status(),
            }

        # /goals
        if path == "/goals":
            # Response: [GoalDict, ...]  ordered by priority ASC (1 = highest)
            # Empty array when no goals exist.
            # See _goal_dict() for the full GoalDict field list.
            return [_goal_dict(g) for g in db.get_all_goals()]

        # /goals/{id}
        if path.startswith("/goals/"):
            parts = path.split("/")  # ['', 'goals', id, ...]
            if len(parts) < 3:
                raise _NotFound("missing goal_id")
            goal_id = parts[2]

            if len(parts) == 3:
                # /goals/{id}
                # Response: GoalDict  — see _goal_dict() for the full field list.
                goal = db.get_goal(goal_id)
                if not goal:
                    raise _NotFound(f"goal {goal_id!r} not found")
                return _goal_dict(goal)

            sub = parts[3] if len(parts) > 3 else ""

            if sub == "sessions":
                # /goals/{id}/sessions?n=20
                # Response: [SessionRow, ...]  most-recent first (by session_id DESC)
                # Each SessionRow:
                #   {
                #     "session_id":   int,         -- auto-increment integer PK
                #     "goal_id":      str,
                #     "status":       str | null,  -- null while session is still open;
                #                                     terminal values: "done", "stuck",
                #                                     "max_actions", "error_streak",
                #                                     "context_overflow", "api_error"
                #     "summary":      str | null,  -- one-paragraph summary once closed
                #     "tick_start":   int,
                #     "tick_end":     int | null,  -- null while still open
                #     "action_count": int | null   -- tool calls executed; null while open
                #   }
                n = _int_qs(qs, "n", 20)
                return db.get_recent_sessions(n=n, goal_id=goal_id)

            if sub == "journals":
                if len(parts) == 4:
                    # /goals/{id}/journals  — index only, no file content
                    # Response: [JournalEntry, ...]  ordered tick_start ASC (oldest first)
                    # Each JournalEntry:
                    #   {
                    #     "goal_id":    str,
                    #     "tick_start": int,
                    #     "tick_end":   int,
                    #     "file_path":  str,  -- relative to workspace root
                    #     "summary":    str   -- first line of the journal entry
                    #   }
                    return db.list_journals_for_goal(goal_id)
                # /goals/{id}/journals/{tick_start}  — full markdown content
                # Response:
                #   {
                #     "file_path": str,   -- relative to workspace root
                #     "content":   str    -- raw Markdown; "" if file is missing on disk
                #   }
                tick_start = parts[4]
                entries = db.list_journals_for_goal(goal_id)
                entry = next(
                    (e for e in entries if str(e["tick_start"]) == tick_start), None
                )
                if not entry:
                    raise _NotFound(f"journal tick_start={tick_start} not found")
                content = store.read_journal_file(entry["file_path"])
                return {"file_path": entry["file_path"], "content": content or ""}

            if sub == "checkpoint":
                # /goals/{id}/checkpoint
                # Response:
                #   {
                #     "content": str  -- full text of CHECKPOINT.md for this goal's workspace;
                #                        "" if the file has not been written yet
                #   }
                goal = db.get_goal(goal_id)
                if not goal:
                    raise _NotFound(f"goal {goal_id!r} not found")
                cp = store.read_checkpoint(goal.workspace_path)
                return {"content": cp or ""}

            raise _NotFound(f"unknown sub-resource {sub!r}")

        # /sessions
        if path == "/sessions":
            # Response: [SessionRow, ...]  across all goals, most-recent first
            # SessionRow shape is identical to /goals/{id}/sessions above.
            n = _int_qs(qs, "n", 20)
            return db.get_recent_sessions(n=n)

        # /ticks
        if path == "/ticks":
            # Response: [TickDict, ...]  oldest-first within the returned window
            # Supports optional ?goal_id=<id> to filter to a single goal.
            # See _tick_dict() for the full TickDict field list.
            n = _int_qs(qs, "n", 50)
            goal_id = _str_qs(qs, "goal_id")
            ticks = db.get_recent_ticks(n=n, goal_id=goal_id)
            return [_tick_dict(t) for t in ticks]

        # /knowledge
        if path == "/knowledge":
            # Response: [KnowledgeEntry, ...]  ordered alphabetically by topic
            # Each KnowledgeEntry:
            #   {
            #     "topic":           str,  -- e.g. "nginx"
            #     "summary":         str,  -- one-line description
            #     "updated_at_tick": int
            #   }
            return db.list_knowledge()

        if path.startswith("/knowledge/"):
            topic = path[len("/knowledge/"):]
            content = db.get_knowledge(topic)
            if content is None:
                raise _NotFound(f"knowledge topic {topic!r} not found")
            # Response:
            #   {
            #     "topic":   str,  -- same as the URL segment
            #     "content": str   -- raw Markdown of the knowledge entry
            #   }
            return {"topic": topic, "content": content}

        # /weeklies
        if path == "/weeklies":
            # Response: [WeeklyEntry, ...]  oldest-first within the returned window
            # Each WeeklyEntry:
            #   {
            #     "tick":      int,
            #     "file_path": str,  -- relative to workspace root
            #     "summary":   str   -- first line of the weekly summary
            #   }
            n = _int_qs(qs, "n", 5)
            return db.get_recent_weeklies(n=n)

        if path.startswith("/weeklies/"):
            tick_str = path[len("/weeklies/"):]
            entries = db.get_recent_weeklies(n=10_000)
            entry = next((e for e in entries if str(e["tick"]) == tick_str), None)
            if not entry:
                raise _NotFound(f"weekly tick={tick_str} not found")
            content = store.read_weekly_file(entry["file_path"])
            # Response:
            #   {
            #     "tick":    int,
            #     "content": str  -- raw Markdown of the weekly summary file;
            #                        "" if file is missing on disk
            #   }
            return {"tick": entry["tick"], "content": content or ""}

        # /outbox
        if path == "/outbox":
            # Response: [OutboxRow, ...]  oldest-first within the returned window
            # Each OutboxRow:
            #   {
            #     "msg_id":   str,    -- UUID string
            #     "reply_to": str,    -- inbox msg_id this is a reply to; "" if unsolicited
            #     "title":    str,
            #     "content":  str,    -- full message body (Markdown)
            #     "sent_at":  float   -- Unix timestamp (time.time())
            #   }
            n = _int_qs(qs, "n", 20)
            return db.get_recent_outbox(n=n)

        # /inbox
        if path == "/inbox":
            # Response: [InboxRow, ...]  oldest-first within the returned window,
            # includes both answered and unanswered messages.
            # Each InboxRow:
            #   {
            #     "msg_id":      str,           -- UUID string (set by POST /inbox)
            #     "text":        str,
            #     "received_at": float,          -- Unix timestamp
            #     "answered_at": float | null    -- null = not yet answered by the Executive
            #   }
            n = _int_qs(qs, "n", 20)
            return db.get_recent_inbox(n=n)

        # /files  or  /files/{rel_path} — serve any workspace file
        if path == "/files" or path.startswith("/files/"):
            rel = path[len("/files/"):] if path.startswith("/files/") else ""
            # Response shape depends on whether the path is a directory or file;
            # see _serve_file() for the exact shapes.
            return self._serve_file(rel)

        # /audit/llm/dates
        if path == "/audit/llm/dates":
            # Response: ["YYYY-MM-DD", ...]  newest date first
            # LLM audit files are retained for 2 days (today + yesterday).
            # Empty array when auditing is disabled.
            if not self.audit:
                return []
            return self.audit.list_llm_dates()

        # /audit/tools/dates
        if path == "/audit/tools/dates":
            # Response: ["YYYY-MM-DD", ...]  newest date first
            # Tool audit files are retained for 31 days.
            # Empty array when auditing is disabled.
            if not self.audit:
                return []
            return self.audit.list_tool_dates()

        # /audit/llm/history
        if path == "/audit/llm/history":
            # Response: [LLMEntry, ...]  — all entries from the JSONL file for the
            # requested date, in the order they were written (chronological).
            # Requires ?date=YYYY-MM-DD.
            # See /audit/llm for the LLMEntry field list.
            # Empty array when auditing is disabled or the date file doesn't exist.
            if not self.audit:
                return []
            date = _str_qs(qs, "date") or ""
            if not date:
                raise _BadRequest("?date=YYYY-MM-DD required")
            return self.audit.read_llm_history(date)

        # /audit/tools/history
        if path == "/audit/tools/history":
            # Response: [ToolEntry, ...]  — all entries from the JSONL file for
            # the requested date, in the order they were written (chronological).
            # Requires ?date=YYYY-MM-DD.
            # See /audit/tools for the ToolEntry field list.
            # Empty array when auditing is disabled or the date file doesn't exist.
            if not self.audit:
                return []
            date = _str_qs(qs, "date") or ""
            if not date:
                raise _BadRequest("?date=YYYY-MM-DD required")
            return self.audit.read_tool_history(date)

        # /audit/llm
        if path == "/audit/llm":
            # Response: [LLMEntry, ...]  from the in-memory ring buffer (last 500),
            # oldest-first within the returned slice.
            # Supports ?n=50 (max entries) and ?since=<unix_ts> (exclude older entries).
            # Each LLMEntry:
            #   {
            #     "ts":         float,        -- Unix timestamp of the call
            #     "actor":      str,          -- "executive" | "worker" | "summarizer"
            #     "tick_id":    int | null,
            #     "session_id": int | null,
            #     "goal_id":    str | null,
            #     "thinking":   str | null,   -- raw model reasoning chain (audit-only)
            #     "content":    str | null,   -- text response (null when tool_calls non-empty)
            #     "tool_calls": [...],        -- list of tool call objects the model requested
            #     "usage":      {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
            #   }
            # Empty array when auditing is disabled.
            if not self.audit:
                return []
            n = _int_qs(qs, "n", 50)
            since = _float_qs(qs, "since", 0.0)
            return self.audit.get_recent_llm(n=n, since=since)

        # /audit/tools
        if path == "/audit/tools":
            # Response: [ToolEntry, ...]  from the in-memory ring buffer (last 500),
            # oldest-first within the returned slice.
            # Supports ?n=100 and ?since=<unix_ts>.
            # Each ToolEntry:
            #   {
            #     "ts":         float,   -- Unix timestamp
            #     "actor":      str,     -- which agent ran the tool
            #     "tick_id":    int | null,
            #     "session_id": int | null,
            #     "goal_id":    str | null,
            #     "tool_name":  str,     -- e.g. "shell", "write_file", "read_file"
            #     "args":       {...},   -- arguments passed to the tool
            #     "output":     str,     -- raw tool output (may be long)
            #     "is_error":   bool,
            #     "truncated":  bool     -- true if output was cut off before sending to agent
            #   }
            # Empty array when auditing is disabled.
            if not self.audit:
                return []
            n = _int_qs(qs, "n", 100)
            since = _float_qs(qs, "since", 0.0)
            return self.audit.get_recent_tools(n=n, since=since)

        raise _NotFound(f"unknown path {path!r}")

    # ── POST router ────────────────────────────────────────────────────────

    def _route_post(self, path: str, body: dict) -> Any:
        if path == "/inbox":
            # Request body: {"text": str}  — plain text message to the agent
            text = str(body.get("text", "")).strip()
            if not text:
                raise _BadRequest("'text' field is required and must not be empty")
            msg_id = str(uuid.uuid4())
            self.db.add_inbox_message(
                msg_id=msg_id,
                text=text,
                received_at=time.time(),
            )
            log.info("Gateway: inbox message %s written: %s", msg_id, text[:80])
            # Response:
            #   {
            #     "msg_id": str,            -- UUID4 assigned to this message;
            #                                  use it to look up the message in GET /inbox
            #     "status": "queued"        -- always "queued" on success
            #   }
            return {"msg_id": msg_id, "status": "queued"}

        raise _NotFound(f"unknown POST path {path!r}")

    # ── File serving ───────────────────────────────────────────────────────

    def _serve_file(self, rel_path: str) -> dict:
        """
        Serve a file (or directory listing) under workspace root.
        Guards against path traversal — any path that resolves outside the
        workspace root is rejected with a 404.

        Directory response:
            {
                "path":    str,          -- rel_path as requested
                "is_dir":  true,
                "entries": [
                    {
                        "name":   str,
                        "is_dir": bool,
                        "size":   int    -- bytes; 0 for sub-directories
                    },
                    ...                  -- sorted alphabetically by name
                ]
            }

        File response:
            {
                "path":      str,        -- rel_path as requested
                "is_dir":    false,
                "truncated": bool,       -- true when the file exceeds 1 MB (_MAX_FILE_BYTES)
                "size":      int,        -- actual file size in bytes (even when truncated)
                "content":   str         -- UTF-8 text; replacement char (U+FFFD) on decode errors
            }
        """
        if not self.workspace:
            raise _NotFound("/files endpoint requires workspace to be configured")
        root = self.workspace.root
        # Resolve and verify the path stays inside workspace root
        try:
            target = (root / rel_path).resolve()
            target.relative_to(root)  # raises ValueError if outside root
        except (ValueError, OSError):
            raise _NotFound(f"path {rel_path!r} is outside workspace root or invalid")

        if not target.exists():
            raise _NotFound(f"{rel_path!r} not found")

        if target.is_dir():
            # Return a directory listing
            entries = [
                {"name": p.name, "is_dir": p.is_dir(), "size": p.stat().st_size if p.is_file() else 0}
                for p in sorted(target.iterdir())
            ]
            return {"path": rel_path, "is_dir": True, "entries": entries}

        # File — read and return content (with size cap)
        size = target.stat().st_size
        if size > _MAX_FILE_BYTES:
            return {
                "path": rel_path,
                "is_dir": False,
                "truncated": True,
                "size": size,
                "content": target.read_bytes()[:_MAX_FILE_BYTES].decode("utf-8", errors="replace"),
            }
        return {
            "path": rel_path,
            "is_dir": False,
            "truncated": False,
            "size": size,
            "content": target.read_text(encoding="utf-8", errors="replace"),
        }

    # ── HTTP helpers ───────────────────────────────────────────────────────

    def _send_json(self, status: int, data: Any) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: N802, A002
        # Route access log through the standard logging system instead of stderr
        log.debug("gateway %s", format % args)


# ── Exceptions ─────────────────────────────────────────────────────────────

class _NotFound(Exception):
    pass


class _BadRequest(Exception):
    pass


# ── Helpers ────────────────────────────────────────────────────────────────

def _goal_dict(goal: Any) -> dict:
    """
    Serialize a Goal model to a plain dict for JSON responses.

    Returned shape:
        {
            "goal_id":          str,   -- e.g. "goal-007"
            "title":            str,
            "description":      str,
            "status":           str,   -- "active" | "blocked" | "done" | "abandoned" | "paused"
            "priority":         int,   -- lower = higher priority (1 = top)
            "created_at_tick":  int,
            "last_worked_tick": int,   -- 0 if the goal has never been worked on
            "total_ticks":      int,   -- cumulative ticks spent on this goal
            "workspace_path":   str,   -- relative to workspace root, e.g. "goals/goal-007-name"
            "blocked_reason":   str    -- human-readable blocker; "" when not blocked
        }
    """
    from .models import Goal
    return {
        "goal_id": goal.goal_id,
        "title": goal.title,
        "description": goal.description,
        "status": goal.status,
        "priority": goal.priority,
        "created_at_tick": goal.created_at_tick,
        "last_worked_tick": goal.last_worked_tick,
        "total_ticks": goal.total_ticks,
        "workspace_path": goal.workspace_path,
        "blocked_reason": goal.blocked_reason,
    }


def _tick_dict(tick: Any) -> dict:
    """
    Serialize a TickRecord model to a plain dict for JSON responses.

    Returned shape:
        {
            "tick_id":     int,
            "session_id":  int | null,  -- null for executive / infra ticks
            "goal_id":     str | null,  -- null for infra ticks
            "actor":       str,         -- "executive" | "worker" | "summarizer" | "infra"
            "action_type": str,         -- e.g. "shell", "write", "decision", "journal"
            "summary":     str,         -- one-line description of what happened
            "outcome":     str          -- "ok" | "error"
        }
    """
    return {
        "tick_id": tick.tick_id,
        "session_id": tick.session_id,
        "goal_id": tick.goal_id,
        "actor": tick.actor,
        "action_type": tick.action_type,
        "summary": tick.summary,
        "outcome": tick.outcome,
    }


def _int_qs(qs: dict, key: str, default: int = 0) -> int:
    """Return the first value of query-string `key` parsed as int, or `default`."""
    vals = qs.get(key)
    if vals:
        try:
            return int(vals[0])
        except (ValueError, IndexError):
            pass
    return default


def _float_qs(qs: dict, key: str, default: float = 0.0) -> float:
    """Return the first value of query-string `key` parsed as float, or `default`."""
    vals = qs.get(key)
    if vals:
        try:
            return float(vals[0])
        except (ValueError, IndexError):
            pass
    return default


def _str_qs(qs: dict, key: str, default: str | None = None) -> str | None:
    """Return the first value of query-string `key` as a string, or `default`."""
    vals = qs.get(key)
    if vals:
        return vals[0]
    return default
