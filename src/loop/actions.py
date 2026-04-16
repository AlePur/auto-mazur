"""
Action executor — carries out the Executive's decisions on the system state.

Each action is a deterministic, synchronous operation.
The only exception is assign_task, which returns a Task that the main loop
will run through the WorkerSession (not executed here).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..db import Database
from ..models import (
    ACTOR_EXECUTIVE,
    EXEC_TOOL_ASSIGN_TASK,
    EXEC_TOOL_CREATE_GOAL,
    EXEC_TOOL_FORGET_KNOWLEDGE,
    EXEC_TOOL_SEND_USER_MESSAGE,
    EXEC_TOOL_UPDATE_GOAL,
    EXEC_TOOL_WRITE_KNOWLEDGE,
    ExecutiveAction,
    GOAL_STATUS_ACTIVE,
    Goal,
    OUTCOME_OK,
    Task,
    TickRecord,
)
from ..store import Store
from ..workspace import Workspace

log = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Outcome of executing one ExecutiveAction."""
    action: ExecutiveAction
    task: Task | None = None             # set only for assign_task
    outbox_entry: dict | None = None     # set only for send_user_message
    error: str | None = None


class ActionExecutor:
    def __init__(self, *, db: Database, workspace: Workspace, store: Store) -> None:
        self._db = db
        self._workspace = workspace
        self._store = store
        self._goal_counter = 0          # incremented when new goals are created

    def execute(self, action: ExecutiveAction) -> ActionResult:
        """Execute one action. Returns an ActionResult."""
        try:
            match action.tool:
                case "assign_task":
                    return self._assign_task(action)
                case "create_goal":
                    return self._create_goal(action)
                case "update_goal":
                    return self._update_goal(action)
                case "send_user_message":
                    return self._send_user_message(action)
                case "write_knowledge":
                    return self._write_knowledge(action)
                case "forget_knowledge":
                    return self._forget_knowledge(action)
                case _:
                    log.warning("Unknown action tool: %r", action.tool)
                    return ActionResult(action=action, error=f"unknown tool: {action.tool}")
        except Exception as exc:
            log.exception("Action %r failed: %s", action.tool, exc)
            return ActionResult(action=action, error=str(exc))

    # ── Individual actions ─────────────────────────────────────────────────

    def _assign_task(self, action: ExecutiveAction) -> ActionResult:
        params = action.params
        goal_id = str(params.get("goal_id", ""))
        description = str(params.get("description", ""))
        criteria = str(params.get("criteria", ""))

        if not goal_id or not description:
            return ActionResult(action=action, error="assign_task: missing goal_id or description")

        goal = self._db.get_goal(goal_id)
        if not goal:
            return ActionResult(action=action, error=f"assign_task: goal {goal_id!r} not found")

        task = Task(
            goal_id=goal_id,
            description=description,
            criteria=criteria,
        )
        log.info("Task assigned for %s: %s", goal_id, description[:80])
        return ActionResult(action=action, task=task)

    def _create_goal(self, action: ExecutiveAction) -> ActionResult:
        params = action.params
        title = str(params.get("title", "untitled"))
        description = str(params.get("description", ""))
        priority = int(params.get("priority", 10))

        # Generate goal ID
        self._goal_counter += 1
        # Read current max goal number from DB to survive restarts
        existing = self._db.get_all_goals()
        max_num = max(
            (int(g.goal_id.split("-")[1]) for g in existing
             if g.goal_id.startswith("goal-") and g.goal_id.split("-")[1].isdigit()),
            default=0
        )
        goal_num = max(max_num + 1, self._goal_counter)
        goal_id = f"goal-{goal_num:03d}"

        # Create workspace directory + store state directory
        workspace_path = self._workspace.create_goal_dir(goal_id, title)
        self._store.create_goal_state_dir(workspace_path)

        # Get current tick (0 if DB is empty — fixed up by main loop)
        current_tick = self._db.get_last_tick_id()

        goal = Goal(
            goal_id=goal_id,
            title=title,
            description=description,
            status=GOAL_STATUS_ACTIVE,
            priority=priority,
            created_at_tick=current_tick,
            last_worked_tick=0,
            total_ticks=0,
            workspace_path=workspace_path,
        )
        self._db.create_goal(goal)
        log.info("Created goal %s: %s (priority %d)", goal_id, title, priority)
        return ActionResult(action=action)

    def _update_goal(self, action: ExecutiveAction) -> ActionResult:
        params = action.params
        goal_id = str(params.get("goal_id", ""))

        if not goal_id:
            return ActionResult(action=action, error="update_goal: missing goal_id")

        updates: dict[str, Any] = {}
        if "title" in params:
            updates["title"] = str(params["title"])
        if "description" in params:
            updates["description"] = str(params["description"])
        if "status" in params:
            updates["status"] = str(params["status"])
        if "priority" in params:
            updates["priority"] = int(params["priority"])
        if "blocked_reason" in params:
            updates["blocked_reason"] = str(params["blocked_reason"])

        if not updates:
            return ActionResult(action=action, error="update_goal: no fields to update")

        self._db.update_goal(goal_id, **updates)
        log.info("Updated goal %s: %s", goal_id, updates)
        return ActionResult(action=action)

    def _send_user_message(self, action: ExecutiveAction) -> ActionResult:
        params = action.params
        title = str(params.get("title", "")).strip()
        content = str(params.get("content", "")).strip()
        re_message_id = str(params.get("re_message_id", "")).strip()

        if not title:
            return ActionResult(action=action, error="send_user_message: missing title")
        if not content:
            return ActionResult(action=action, error="send_user_message: missing content")

        entry = {
            "title": title,
            "content": content,
            "re_message_id": re_message_id,
        }
        log.info(
            "Executive → user | title=%r re=%r: %s",
            title, re_message_id or "(unprompted)", content[:80],
        )
        return ActionResult(action=action, outbox_entry=entry)

    def _write_knowledge(self, action: ExecutiveAction) -> ActionResult:
        params = action.params
        topic = str(params.get("topic", "")).strip()
        content = str(params.get("content", "")).strip()

        if not topic:
            return ActionResult(action=action, error="write_knowledge: missing topic")
        if not content:
            return ActionResult(action=action, error="write_knowledge: missing content")

        # Extract first meaningful line as summary
        summary_line = next(
            (
                line.strip()
                for line in content.splitlines()
                if line.strip() and not line.startswith("#")
            ),
            "",
        )[:120]

        current_tick = self._db.get_last_tick_id()
        self._db.upsert_knowledge(
            topic=topic,
            content=content,
            summary=summary_line,
            tick=current_tick,
        )
        log.info("Executive wrote knowledge: topic=%r (%d chars)", topic, len(content))
        return ActionResult(action=action)

    def _forget_knowledge(self, action: ExecutiveAction) -> ActionResult:
        topic = str(action.params.get("topic", "")).strip()
        if not topic:
            return ActionResult(action=action, error="forget_knowledge: missing topic")

        deleted = self._db.delete_knowledge(topic)
        if deleted:
            log.info("Executive deleted knowledge: topic=%r", topic)
        else:
            log.warning("forget_knowledge: topic %r not found", topic)
            return ActionResult(action=action, error=f"forget_knowledge: topic {topic!r} not found")
        return ActionResult(action=action)
