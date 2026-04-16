"""
Executive character — the strategic decision-maker.

The Executive is the only character that:
  - Knows about all goals
  - Knows about the user inbox/outbox
  - Makes resource-allocation decisions
  - Writes and manages knowledge
  - Requests journaling via the Summarizer

It operates in a two-phase loop each tick:
  1. Query phase — call read-only query tools to gather more detail
  2. Decision phase — call action tools to produce decisions

Query tools never mutate state; action tools always do.
The infrastructure runs the query loop until either an action tool is called
or max_executive_queries is reached, then executes the actions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..models import ExecutiveAction, EXEC_TOOLS

log = logging.getLogger(__name__)

# ── Tool name constants ────────────────────────────────────────────────────

# Query tools (read-only, no state mutation)
EXEC_QUERY_READ_JOURNAL      = "read_journal"
EXEC_QUERY_READ_KNOWLEDGE    = "read_knowledge"
EXEC_QUERY_SEARCH_KNOWLEDGE  = "search_knowledge"
EXEC_QUERY_LIST_SESSIONS     = "list_sessions"
EXEC_QUERY_READ_FILE         = "read_file"

EXEC_QUERY_TOOLS = {
    EXEC_QUERY_READ_JOURNAL,
    EXEC_QUERY_READ_KNOWLEDGE,
    EXEC_QUERY_SEARCH_KNOWLEDGE,
    EXEC_QUERY_LIST_SESSIONS,
    EXEC_QUERY_READ_FILE,
}

# ── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (Path(__file__).parent.parent / "souls" / "executive.md").read_text()

# ── Query tool schemas ─────────────────────────────────────────────────────

QUERY_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_journal",
            "description": "Read recent journal entries for a specific goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "string",
                        "description": "ID of the goal.",
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of recent journal entries to read (default 3, max 10).",
                    },
                },
                "required": ["goal_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_knowledge",
            "description": "Read the full content of a knowledge entry by topic name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The knowledge topic, e.g. 'nginx'.",
                    }
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Full-text search across all knowledge entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword query to search for.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sessions",
            "description": "List recent Worker sessions for a goal, with status and summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "string",
                        "description": "ID of the goal.",
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of recent sessions to return (default 5, max 20).",
                    },
                },
                "required": ["goal_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the workspace. "
                "Paths are relative to the workspace root (e.g. 'goals/goal-001-slug/src/main.py'). "
                "Absolute paths are also accepted. "
                "By default returns the first 100 lines. "
                "Use the `lines` parameter to read a specific range, e.g. '100-200'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to workspace root or absolute).",
                    },
                    "lines": {
                        "type": "string",
                        "description": (
                            "Line range to return, zero-indexed, inclusive. "
                            "Format: 'START-END', e.g. '0-100', '200-300'. "
                            "Omit for default first 100 lines."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
]

# ── Action tool schemas ────────────────────────────────────────────────────

ACTION_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "assign_task",
            "description": (
                "Assign a concrete task to the Worker for a goal. "
                "The Worker executes shell commands and file operations "
                "until the task is done or it gets stuck. "
                "Goals are long-running projects — assign incremental tasks "
                "and continue working on the same goal across many sessions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "string",
                        "description": "ID of the goal this task belongs to.",
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Concrete, actionable description of exactly what to do. "
                            "Be specific. The Worker has no other context."
                        ),
                    },
                    "criteria": {
                        "type": "string",
                        "description": (
                            "How to verify the task is done. "
                            "E.g. 'curl localhost:80 returns 200' or "
                            "'the test suite passes with zero failures'."
                        ),
                    },
                },
                "required": ["goal_id", "description", "criteria"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_goal",
            "description": (
                "Create a new long-running goal with a workspace directory. "
                "Goals represent ongoing projects or objectives that may require "
                "many sessions over time to complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short, human-readable name (used as directory slug).",
                    },
                    "description": {
                        "type": "string",
                        "description": "Full description of what this goal is about.",
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority level; lower number = higher priority. 1 = top.",
                    },
                },
                "required": ["title", "description", "priority"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_goal",
            "description": (
                "Update a goal's title, description, status, priority, or blocked_reason. "
                "Use status='done' only when the full goal objective is achieved — "
                "not just because one session completed. "
                "Use 'blocked' when waiting for something external, "
                "'paused' to defer it, 'abandoned' to drop it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {"type": "string"},
                    "title": {
                        "type": "string",
                        "description": "New title for the goal.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Updated description of what this goal is about.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "blocked", "paused", "done", "abandoned"],
                    },
                    "priority": {"type": "integer"},
                    "blocked_reason": {
                        "type": "string",
                        "description": "Required when status=blocked.",
                    },
                },
                "required": ["goal_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_user_message",
            "description": (
                "Send a message to the user. "
                "Every message has a title (subject line) and content (body). "
                "Optionally set re_message_id to quote the inbox message you are replying to — "
                "this also marks that inbox message as answered. "
                "You can send multiple messages per tick, and you can follow up on "
                "already-answered messages at any time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short subject line / headline for the message.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full body of the message. Markdown is fine.",
                    },
                    "re_message_id": {
                        "type": "string",
                        "description": (
                            "Optional. ID of the inbox message this is a reply to. "
                            "Setting this marks that inbox message as answered."
                        ),
                    },
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_journaling",
            "description": (
                "Request the Summarizer to write a journal entry for a goal. "
                "Use this after a productive run of sessions, when you want to "
                "capture progress before the goal's automatic journal threshold is reached, "
                "or before a long break from working on a goal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "string",
                        "description": "ID of the goal to journal.",
                    }
                },
                "required": ["goal_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_knowledge",
            "description": (
                "Write or update a knowledge entry. "
                "Use this to record important facts, system details, learned patterns, "
                "or reusable information that will be useful for future Worker sessions. "
                "Knowledge is automatically surfaced to Workers via keyword search. "
                "Content should be Markdown, with a clear heading."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": (
                            "Short identifier for the topic, e.g. 'nginx', 'postgres-backups', "
                            "'deployment-process'. Used as the lookup key."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "Full Markdown content of the knowledge entry.",
                    },
                },
                "required": ["topic", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_knowledge",
            "description": (
                "Delete a knowledge entry permanently. "
                "Use this when information is outdated, incorrect, or no longer relevant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic key to delete.",
                    }
                },
                "required": ["topic"],
            },
        },
    },
]

# Combined schema list: queries first, then actions
ALL_TOOL_SCHEMAS: list[dict[str, Any]] = QUERY_TOOL_SCHEMAS + ACTION_TOOL_SCHEMAS

# ── Response parsing ───────────────────────────────────────────────────────

def parse_actions(tool_calls: list) -> list[ExecutiveAction]:
    """
    Convert LLM tool calls into ExecutiveAction objects.
    Only processes action tools; query tools are handled by the loop.
    Unknown tool names are logged and skipped.
    """
    actions: list[ExecutiveAction] = []
    for tc in tool_calls:
        if tc.name not in EXEC_TOOLS:
            log.warning("Executive called unknown tool %r — skipping", tc.name)
            continue
        actions.append(ExecutiveAction(tool=tc.name, params=tc.arguments))
    return actions
