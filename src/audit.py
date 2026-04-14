"""
AuditLogger — append-only structured audit trail for every LLM call and tool
execution.

Two log streams, each written as JSONL to a daily rotating file:

  {workspace}/audit/llm/YYYY-MM-DD.jsonl
      One entry per LLM call (thinking + content + tool_calls + usage).
      Retention: 2 days (today + yesterday).  Oldest files are removed on each
      write so disk usage stays bounded.

  {workspace}/audit/tools/YYYY-MM-DD.jsonl
      One entry per worker tool execution.
      Retention: 31 days.

Both streams also keep an in-memory ring buffer (last 500 entries each) for
fast polling by the gateway HTTP server without hitting disk.

Thread safety: a single threading.Lock guards both the in-memory buffers and
the file writes so that concurrent HTTP reads and agent writes do not race.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_RING_MAXLEN = 500
_LLM_RETENTION_DAYS = 2      # keep today + yesterday
_TOOL_RETENTION_DAYS = 31


class AuditLogger:
    """
    Structured audit logger for LLM outputs and tool executions.

    Parameters
    ----------
    workspace_root:
        Root of the agent workspace.  Audit logs are written under
        ``{workspace_root}/audit/``.
    """

    def __init__(self, workspace_root: str | Path) -> None:
        self._root = Path(workspace_root).resolve()
        self._llm_dir = self._root / "audit" / "llm"
        self._tool_dir = self._root / "audit" / "tools"
        self._llm_dir.mkdir(parents=True, exist_ok=True)
        self._tool_dir.mkdir(parents=True, exist_ok=True)

        # In-memory ring buffers
        self._llm_ring: deque[dict[str, Any]] = deque(maxlen=_RING_MAXLEN)
        self._tool_ring: deque[dict[str, Any]] = deque(maxlen=_RING_MAXLEN)

        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────

    def log_llm(
        self,
        *,
        actor: str,
        tick_id: int | None,
        session_id: int | None,
        goal_id: str | None,
        thinking: str | None,
        content: str | None,
        tool_calls: list[dict[str, Any]],
        usage: dict[str, int],
    ) -> None:
        """
        Record one LLM call result.

        ``thinking`` is the raw reasoning chain from the model (Gemma 4
        reasoning_content).  It is stored here for human audit only — it is
        never forwarded back to any agent context.
        """
        entry: dict[str, Any] = {
            "ts": time.time(),
            "actor": actor,
            "tick_id": tick_id,
            "session_id": session_id,
            "goal_id": goal_id,
            "thinking": thinking,
            "content": content,
            "tool_calls": tool_calls,
            "usage": usage,
        }
        with self._lock:
            self._llm_ring.append(entry)
            self._append_jsonl(self._llm_dir, entry)
            self._rotate(self._llm_dir, _LLM_RETENTION_DAYS)

    def log_tool(
        self,
        *,
        actor: str,
        tick_id: int | None,
        session_id: int | None,
        goal_id: str | None,
        tool_name: str,
        args: dict[str, Any],
        output: str,
        is_error: bool,
        truncated: bool = False,
    ) -> None:
        """Record one tool execution."""
        entry: dict[str, Any] = {
            "ts": time.time(),
            "actor": actor,
            "tick_id": tick_id,
            "session_id": session_id,
            "goal_id": goal_id,
            "tool_name": tool_name,
            "args": args,
            "output": output,
            "is_error": is_error,
            "truncated": truncated,
        }
        with self._lock:
            self._tool_ring.append(entry)
            self._append_jsonl(self._tool_dir, entry)
            self._rotate(self._tool_dir, _TOOL_RETENTION_DAYS)

    # ── Ring-buffer accessors (used by gateway) ────────────────────────────

    def get_recent_llm(self, n: int = 50, since: float = 0.0) -> list[dict[str, Any]]:
        """
        Return up to *n* most-recent LLM entries from the ring buffer,
        optionally filtered to entries with ``ts > since``.
        """
        with self._lock:
            items = list(self._llm_ring)
        if since:
            items = [e for e in items if e["ts"] > since]
        return items[-n:]

    def get_recent_tools(self, n: int = 100, since: float = 0.0) -> list[dict[str, Any]]:
        """
        Return up to *n* most-recent tool entries from the ring buffer,
        optionally filtered to entries with ``ts > since``.
        """
        with self._lock:
            items = list(self._tool_ring)
        if since:
            items = [e for e in items if e["ts"] > since]
        return items[-n:]

    # ── JSONL file helpers ─────────────────────────────────────────────────

    @staticmethod
    def _today_filename() -> str:
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d") + ".jsonl"

    @staticmethod
    def _append_jsonl(directory: Path, entry: dict[str, Any]) -> None:
        path = directory / AuditLogger._today_filename()
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.error("AuditLogger: failed to write %s: %s", path, exc)

    @staticmethod
    def _rotate(directory: Path, keep_days: int) -> None:
        """Delete JSONL files older than *keep_days* days."""
        files = sorted(directory.glob("*.jsonl"))
        # Keep only the most-recent keep_days files
        to_delete = files[: max(0, len(files) - keep_days)]
        for f in to_delete:
            try:
                f.unlink()
                log.debug("AuditLogger: rotated out %s", f)
            except OSError as exc:
                log.warning("AuditLogger: could not delete %s: %s", f, exc)

    # ── History reader (used by gateway /audit/.../history) ───────────────

    def read_llm_history(self, date: str) -> list[dict[str, Any]]:
        """
        Read LLM audit entries for a specific date (``YYYY-MM-DD``).
        Returns an empty list if the file does not exist or cannot be read.
        """
        return self._read_jsonl(self._llm_dir / f"{date}.jsonl")

    def read_tool_history(self, date: str) -> list[dict[str, Any]]:
        """
        Read tool audit entries for a specific date (``YYYY-MM-DD``).
        Returns an empty list if the file does not exist or cannot be read.
        """
        return self._read_jsonl(self._tool_dir / f"{date}.jsonl")

    def list_llm_dates(self) -> list[str]:
        """Return available LLM audit dates (YYYY-MM-DD), newest first."""
        return _list_dates(self._llm_dir)

    def list_tool_dates(self) -> list[str]:
        """Return available tool audit dates (YYYY-MM-DD), newest first."""
        return _list_dates(self._tool_dir)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        entries = []
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError as exc:
            log.warning("AuditLogger: cannot read %s: %s", path, exc)
        return entries


# ── Module-level helpers ───────────────────────────────────────────────────

def _list_dates(directory: Path) -> list[str]:
    files = sorted(directory.glob("*.jsonl"), reverse=True)
    return [f.stem for f in files]
