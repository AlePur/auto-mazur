"""
Worker character — the execution engine.

The Worker:
  - Sees a task description + context, nothing else
  - Has 4 tools: shell, read, write, finish
  - Works freely in a multi-turn conversation until the task is done
  - Never knows about goals, priorities, the user, or other goals

The system prompt is intentionally minimal. The real context comes from
the task briefing assembled by context/worker.py.
"""

from __future__ import annotations

from pathlib import Path

# The Worker's tool schemas are in tools.py (WORKER_TOOL_SCHEMAS).
# We re-export them here so callers can import from characters.worker.
from ..tools import WORKER_TOOL_SCHEMAS  # noqa: F401


SYSTEM_PROMPT = (Path(__file__).parent.parent / "souls" / "worker.md").read_text()
