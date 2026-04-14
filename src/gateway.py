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
  GET /reflections?n=10                   recent reflection index entries
  GET /reflections/{tick}                 reflection markdown content
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
        workspace: "Workspace",
        audit: "AuditLogger | None",
        loop: Any,  # MainLoop — imported lazily to avoid circular dependency
    ) -> None:
        self._host = host
        self._port = port

        # Inject shared state into the handler class.  Because BaseHTTP-
        # RequestHandler is instantiated fresh per request, we use class-level
        # attributes rather than instance attributes.
        _Handler.db = db
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
    workspace: "Workspace"
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

    def _route_get(self, path: str, qs: dict) -> Any:  # noqa: C901
        db = self.db
        ws = self.workspace

        # /status
        if path == "/status":
            return {
                "tick": db.get_last_tick_id(),
                "goal_counts": db.get_goal_counts_by_status(),
            }

        # /goals
        if path == "/goals":
            return [_goal_dict(g) for g in db.get_all_goals()]

        # /goals/{id}
        if path.startswith("/goals/"):
            parts = path.split("/")  # ['', 'goals', id, ...]
            if len(parts) < 3:
                raise _NotFound("missing goal_id")
            goal_id = parts[2]

            if len(parts) == 3:
                # /goals/{id}
                goal = db.get_goal(goal_id)
                if not goal:
                    raise _NotFound(f"goal {goal_id!r} not found")
                return _goal_dict(goal)

            sub = parts[3] if len(parts) > 3 else ""

            if sub == "sessions":
                n = _int_qs(qs, "n", 20)
                return db.get_recent_sessions(n=n, goal_id=goal_id)

            if sub == "journals":
                if len(parts) == 4:
                    return db.list_journals_for_goal(goal_id)
                tick_start = parts[4]
                entries = db.list_journals_for_goal(goal_id)
                entry = next(
                    (e for e in entries if str(e["tick_start"]) == tick_start), None
                )
                if not entry:
                    raise _NotFound(f"journal tick_start={tick_start} not found")
                content = ws.read_journal_file(entry["file_path"])
                return {"file_path": entry["file_path"], "content": content or ""}

            if sub == "checkpoint":
                goal = db.get_goal(goal_id)
                if not goal:
                    raise _NotFound(f"goal {goal_id!r} not found")
                cp = ws.read_checkpoint(goal.workspace_path)
                return {"content": cp or ""}

            raise _NotFound(f"unknown sub-resource {sub!r}")

        # /sessions
        if path == "/sessions":
            n = _int_qs(qs, "n", 20)
            return db.get_recent_sessions(n=n)

        # /ticks
        if path == "/ticks":
            n = _int_qs(qs, "n", 50)
            goal_id = _str_qs(qs, "goal_id")
            ticks = db.get_recent_ticks(n=n, goal_id=goal_id)
            return [_tick_dict(t) for t in ticks]

        # /knowledge
        if path == "/knowledge":
            return db.list_knowledge()

        if path.startswith("/knowledge/"):
            topic = path[len("/knowledge/"):]
            content = ws.read_knowledge(topic)
            if content is None:
                raise _NotFound(f"knowledge topic {topic!r} not found")
            return {"topic": topic, "content": content}

        # /reflections
        if path == "/reflections":
            n = _int_qs(qs, "n", 10)
            return db.get_recent_reflections(n=n)

        if path.startswith("/reflections/"):
            tick_str = path[len("/reflections/"):]
            entries = db.get_recent_reflections(n=10_000)
            entry = next((e for e in entries if str(e["tick"]) == tick_str), None)
            if not entry:
                raise _NotFound(f"reflection tick={tick_str} not found")
            content = ws.read_reflection_file(entry["file_path"])
            return {"tick": entry["tick"], "content": content or ""}

        # /weeklies
        if path == "/weeklies":
            n = _int_qs(qs, "n", 5)
            return db.get_recent_weeklies(n=n)

        if path.startswith("/weeklies/"):
            tick_str = path[len("/weeklies/"):]
            entries = db.get_recent_weeklies(n=10_000)
            entry = next((e for e in entries if str(e["tick"]) == tick_str), None)
            if not entry:
                raise _NotFound(f"weekly tick={tick_str} not found")
            content = ws.read_weekly_file(entry["file_path"])
            return {"tick": entry["tick"], "content": content or ""}

        # /outbox
        if path == "/outbox":
            n = _int_qs(qs, "n", 20)
            return db.get_recent_outbox(n=n)

        # /inbox
        if path == "/inbox":
            n = _int_qs(qs, "n", 20)
            return db.get_recent_inbox(n=n)

        # /files/{rel_path} — serve any workspace file
        if path.startswith("/files/"):
            rel = path[len("/files/"):]
            return self._serve_file(rel)

        # /audit/llm/dates
        if path == "/audit/llm/dates":
            if not self.audit:
                return []
            return self.audit.list_llm_dates()

        # /audit/tools/dates
        if path == "/audit/tools/dates":
            if not self.audit:
                return []
            return self.audit.list_tool_dates()

        # /audit/llm/history
        if path == "/audit/llm/history":
            if not self.audit:
                return []
            date = _str_qs(qs, "date") or ""
            if not date:
                raise _BadRequest("?date=YYYY-MM-DD required")
            return self.audit.read_llm_history(date)

        # /audit/tools/history
        if path == "/audit/tools/history":
            if not self.audit:
                return []
            date = _str_qs(qs, "date") or ""
            if not date:
                raise _BadRequest("?date=YYYY-MM-DD required")
            return self.audit.read_tool_history(date)

        # /audit/llm
        if path == "/audit/llm":
            if not self.audit:
                return []
            n = _int_qs(qs, "n", 50)
            since = _float_qs(qs, "since", 0.0)
            return self.audit.get_recent_llm(n=n, since=since)

        # /audit/tools
        if path == "/audit/tools":
            if not self.audit:
                return []
            n = _int_qs(qs, "n", 100)
            since = _float_qs(qs, "since", 0.0)
            return self.audit.get_recent_tools(n=n, since=since)

        raise _NotFound(f"unknown path {path!r}")

    # ── POST router ────────────────────────────────────────────────────────

    def _route_post(self, path: str, body: dict) -> Any:
        if path == "/inbox":
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
            return {"msg_id": msg_id, "status": "queued"}

        raise _NotFound(f"unknown POST path {path!r}")

    # ── File serving ───────────────────────────────────────────────────────

    def _serve_file(self, rel_path: str) -> dict:
        """
        Serve a file under workspace root.  Guards against path traversal.
        """
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

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
        # Route access log through the standard logging system instead of stderr
        log.debug("gateway %s", fmt % args)


# ── Exceptions ─────────────────────────────────────────────────────────────

class _NotFound(Exception):
    pass


class _BadRequest(Exception):
    pass


# ── Helpers ────────────────────────────────────────────────────────────────

def _goal_dict(goal: Any) -> dict:
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
    vals = qs.get(key)
    if vals:
        try:
            return int(vals[0])
        except (ValueError, IndexError):
            pass
    return default


def _float_qs(qs: dict, key: str, default: float = 0.0) -> float:
    vals = qs.get(key)
    if vals:
        try:
            return float(vals[0])
        except (ValueError, IndexError):
            pass
    return default


def _str_qs(qs: dict, key: str, default: str | None = None) -> str | None:
    vals = qs.get(key)
    if vals:
        return vals[0]
    return default
