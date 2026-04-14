"""
Executive character — the strategic decision-maker.

The Executive is the only character that:
  - Knows about all goals
  - Knows about the user inbox/outbox
  - Makes resource-allocation decisions
  - Knows about health issues and reflector observations

It operates in a two-phase loop each tick:
  1. Query phase — call read-only query tools to gather more detail
  2. Decision phase — call action tools to produce decisions

Query tools never mutate state; action tools always do.
The infrastructure runs the query loop until either an action tool is called
or max_executive_queries is reached, then executes the actions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..models import ExecutiveAction, EXEC_TOOLS

log = logging.getLogger(__name__)

# ── Tool name constants ────────────────────────────────────────────────────

# Query tools (read-only, no state mutation)
EXEC_QUERY_READ_CHECKPOINT = "read_checkpoint"
EXEC_QUERY_READ_JOURNAL    = "read_journal"
EXEC_QUERY_READ_KNOWLEDGE  = "read_knowledge"
EXEC_QUERY_SEARCH_KNOWLEDGE = "search_knowledge"
EXEC_QUERY_LIST_SESSIONS   = "list_sessions"
EXEC_QUERY_READ_REFLECTION = "read_reflection"

EXEC_QUERY_TOOLS = {
    EXEC_QUERY_READ_CHECKPOINT,
    EXEC_QUERY_READ_JOURNAL,
    EXEC_QUERY_READ_KNOWLEDGE,
    EXEC_QUERY_SEARCH_KNOWLEDGE,
    EXEC_QUERY_LIST_SESSIONS,
    EXEC_QUERY_READ_REFLECTION,
}

# ── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (Path(__file__).parent.parent / "souls" / "executive.md").read_text()

# ── Query tool schemas ─────────────────────────────────────────────────────

QUERY_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_checkpoint",
            "description": (
                "Read the full CHECKPOINT.md for a goal. "
                "Use this when the truncated checkpoint in the briefing is not enough."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "string",
                        "description": "ID of the goal to read the checkpoint for.",
                    }
                },
                "required": ["goal_id"],
            },
        },
    },
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
            "description": "Read the full content of a knowledge file by topic name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The knowledge topic (file stem), e.g. 'nginx'.",
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
            "description": "Full-text search across all knowledge files.",
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
            "name": "read_reflection",
            "description": (
                "Read the content of recent reflection entries. "
                "Useful when you need the full strategic analysis from a past reflection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of recent reflections to read (default 2, max 5).",
                    }
                },
                "required": [],
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
                "Assign a concrete task to the Worker. "
                "The Worker will execute shell commands and file operations "
                "until the task is done or it gets stuck."
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
            "description": "Create a new goal with a workspace directory.",
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
                "Update a goal's status, priority, or blocked_reason. "
                "Use status='done' when a goal is fully complete, "
                "'blocked' when waiting for something external, "
                "'paused' to defer it, 'abandoned' to drop it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {"type": "string"},
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
            "name": "request_reflection",
            "description": (
                "Trigger a Reflector pass. Use this when: priorities seem off, "
                "after a long run of failures, after completing a major goal, "
                "or when you feel lost. "
                "The Reflector will update priorities and knowledge files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why you're requesting reflection.",
                    }
                },
                "required": ["reason"],
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
