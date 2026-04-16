"""
Domain models — pure data, no logic.

All modules import from here. Nothing here imports from the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Goals ─────────────────────────────────────────────────────────────────

GOAL_STATUS_ACTIVE = "active"
GOAL_STATUS_BLOCKED = "blocked"
GOAL_STATUS_DONE = "done"
GOAL_STATUS_ABANDONED = "abandoned"
GOAL_STATUS_PAUSED = "paused"

GOAL_STATUSES = {
    GOAL_STATUS_ACTIVE,
    GOAL_STATUS_BLOCKED,
    GOAL_STATUS_DONE,
    GOAL_STATUS_ABANDONED,
    GOAL_STATUS_PAUSED,
}


@dataclass
class Goal:
    goal_id: str                # e.g. "goal-007"
    title: str                  # short human name
    description: str            # what this goal is about
    status: str                 # one of GOAL_STATUSES
    priority: int               # lower = higher priority (1 = top)
    created_at_tick: int
    last_worked_tick: int       # 0 if never worked on
    total_ticks: int            # cumulative ticks spent on this goal
    workspace_path: str         # path relative to workspace root, e.g. "goals/goal-007-name"
    blocked_reason: str = ""    # human-readable blocker when status == blocked


# ── Tasks ─────────────────────────────────────────────────────────────────

@dataclass
class Task:
    goal_id: str
    description: str    # concrete, actionable description for the Worker
    criteria: str       # how to know the task is done (verified by Worker via finish())


# ── Sessions ──────────────────────────────────────────────────────────────

SESSION_STATUS_DONE = "done"
SESSION_STATUS_STUCK = "stuck"
SESSION_STATUS_MAX_ACTIONS = "max_actions"
SESSION_STATUS_ERROR_STREAK = "error_streak"
SESSION_STATUS_CONTEXT_OVERFLOW = "context_overflow"
SESSION_STATUS_API_ERROR = "api_error"

SESSION_TERMINAL_STATUSES = {
    SESSION_STATUS_DONE,
    SESSION_STATUS_STUCK,
    SESSION_STATUS_MAX_ACTIONS,
    SESSION_STATUS_ERROR_STREAK,
    SESSION_STATUS_CONTEXT_OVERFLOW,
    SESSION_STATUS_API_ERROR,
}


@dataclass
class SessionResult:
    session_id: int
    goal_id: str
    task: Task
    status: str             # one of SESSION_STATUS_*
    summary: str            # one paragraph; produced by Summarizer or Worker finish()
    tick_start: int
    tick_end: int
    action_count: int       # tool calls executed
    tokens_used: int
    transcript_path: str    # absolute path to the .jsonl transcript on disk


# ── Ticks ─────────────────────────────────────────────────────────────────

ACTOR_EXECUTIVE = "executive"
ACTOR_WORKER = "worker"
ACTOR_SUMMARIZER = "summarizer"
ACTOR_INFRA = "infra"

OUTCOME_OK = "ok"
OUTCOME_ERROR = "error"


@dataclass
class TickRecord:
    tick_id: int
    actor: str              # ACTOR_*
    action_type: str        # e.g. "shell", "write", "decision", "journal"
    summary: str            # ONE line — what happened
    outcome: str            # OUTCOME_OK | OUTCOME_ERROR
    session_id: int | None = None
    goal_id: str | None = None


# ── Executive actions ─────────────────────────────────────────────────────

@dataclass
class ExecutiveAction:
    """
    A single action produced by the Executive character during one tick.
    The Executive may produce multiple actions per tick; the infrastructure
    executes them in order.
    """
    tool: str       # one of the executive tool names below
    params: dict[str, Any] = field(default_factory=dict)


# Executive tool names (mirrors the function-calling schema in characters/executive.py)
EXEC_TOOL_ASSIGN_TASK         = "assign_task"
EXEC_TOOL_CREATE_GOAL         = "create_goal"
EXEC_TOOL_UPDATE_GOAL         = "update_goal"
EXEC_TOOL_SEND_USER_MESSAGE   = "send_user_message"
EXEC_TOOL_REQUEST_JOURNALING  = "request_journaling"
EXEC_TOOL_WRITE_KNOWLEDGE     = "write_knowledge"
EXEC_TOOL_FORGET_KNOWLEDGE    = "forget_knowledge"

EXEC_TOOLS = {
    EXEC_TOOL_ASSIGN_TASK,
    EXEC_TOOL_CREATE_GOAL,
    EXEC_TOOL_UPDATE_GOAL,
    EXEC_TOOL_SEND_USER_MESSAGE,
    EXEC_TOOL_REQUEST_JOURNALING,
    EXEC_TOOL_WRITE_KNOWLEDGE,
    EXEC_TOOL_FORGET_KNOWLEDGE,
}


# ── Health ────────────────────────────────────────────────────────────────

HEALTH_STUCK_LOOP = "stuck_loop"
HEALTH_FAILURE_STREAK = "failure_streak"
HEALTH_NEGLECTED_GOAL = "neglected_goal"


@dataclass
class HealthIssue:
    kind: str       # HEALTH_*
    details: str    # human-readable description injected into next briefing


# ── LLM primitives ────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    content: str | None             # text content (if no tool calls)
    tool_calls: list[ToolCall]      # may be empty
    usage: Usage
    raw: Any = field(repr=False)    # original API response object
    thinking: str | None = None     # reasoning_content from vLLM — audit-only, never forwarded to agents


# ── Worker tool result ────────────────────────────────────────────────────

@dataclass
class ToolResult:
    output: str
    is_error: bool
    truncated: bool
