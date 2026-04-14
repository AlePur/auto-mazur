"""
Executive character — the strategic decision-maker.

The Executive is the only character that:
  - Knows about all goals
  - Knows about the user inbox/outbox
  - Makes resource-allocation decisions
  - Knows about health issues and reflector observations

It does NOT execute any tools itself.  It produces a list of actions that
the infrastructure (loop/actions.py) carries out.

The Executive may produce MULTIPLE actions per tick (e.g., respond to a
user message AND create a new goal AND assign a task — all in one call).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..models import ExecutiveAction, EXEC_TOOLS

log = logging.getLogger(__name__)


# ── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the Executive — the strategic layer of an autonomous agent running \
on a computer.

Your role:
- Decide what to work on
- Create and prioritise goals
- Assign concrete tasks to the Worker
- Respond to user messages (asynchronously — they do not expect instant replies)
- Request reflection when needed

You do NOT execute code or run commands yourself. You delegate work to the \
Worker by calling assign_task.

Operating principles:
- Always be productive. If no urgent work exists, find lower-priority work, \
  do maintenance, or request reflection to reassess.
- Prefer small, concrete tasks over vague large ones.
- When a user message creates a new need, create a goal and/or respond.
- If a Worker session ended with 'stuck', decide whether to retry with a \
  different approach, break the task smaller, or mark the goal blocked.
- If a Worker session ended with 'max_actions' or 'context_overflow', \
  continue the same task — work was partial.

You can call multiple tools in a single response. They are executed in order.
"""

# ── Tool schemas ──────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
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
            "name": "respond",
            "description": (
                "Write a response to a user message. "
                "The response is placed in the outbox and delivered by the gateway. "
                "You can respond to a message multiple times "
                "(e.g. 'working on it' then later 'done')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "ID of the inbox message being responded to.",
                    },
                    "text": {
                        "type": "string",
                        "description": "The response text.",
                    },
                },
                "required": ["message_id", "text"],
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


# ── Response parsing ───────────────────────────────────────────────────────

def parse_actions(tool_calls: list) -> list[ExecutiveAction]:
    """
    Convert LLM tool calls into ExecutiveAction objects.
    Unknown tool names are logged and skipped.
    """
    actions: list[ExecutiveAction] = []
    for tc in tool_calls:
        if tc.name not in EXEC_TOOLS:
            log.warning("Executive called unknown tool %r — skipping", tc.name)
            continue
        actions.append(ExecutiveAction(tool=tc.name, params=tc.arguments))
    return actions
