"""
MainLoop — the top-level forever loop.

Orchestrates:
  1. Health check
  2. Executive tick → list of actions
  3. Execute actions (goal creation, task assignment, responses, reflection)
  4. If a task was assigned: run a WorkerSession
  5. Post-session state updates (goal tick counts, checkpoint)
  6. Consolidation (journal, archive, reflection — based on tick schedule)
  7. Increment tick counter and repeat

The tick counter is the system clock.  It is persisted to the DB after
every action (worker tool call, executive decision) so a crash loses at
most one tick.

Inbox / outbox:
  The gateway integration is intentionally left as a hook.
  By default the inbox is empty and the outbox is logged.
  To wire in real messages: subclass MainLoop and override
  _load_inbox() and _deliver_outbox().
"""

from __future__ import annotations

import logging
import traceback

from ..config import Config
from ..consolidation import Consolidation
from ..context import worker as worker_context
from ..db import Database
from ..health import HealthChecker
from ..llm import LLMClient
from .actions import ActionExecutor, ActionResult
from .executive import ExecutiveTick
from .session import WorkerSession
from ..models import (
    ACTOR_INFRA,
    OUTCOME_OK,
    SessionResult,
    Task,
    TickRecord,
)
from ..tools import ToolExecutor
from ..workspace import Workspace

log = logging.getLogger(__name__)


class MainLoop:
    def __init__(self, config: Config) -> None:
        self._config = config

        # Storage
        self._db = Database(config.db_file())
        self._workspace = Workspace(config.workspace_path())

        # LLM + tools
        self._llm = LLMClient(config)
        self._tools = ToolExecutor(config)

        # Sub-systems
        self._health = HealthChecker(config, self._db)
        self._consolidation = Consolidation(
            config=config,
            db=self._db,
            workspace=self._workspace,
            llm=self._llm,
        )
        self._executive_tick = ExecutiveTick(
            config=config,
            llm=self._llm,
            db=self._db,
            workspace=self._workspace,
        )
        self._action_executor = ActionExecutor(
            db=self._db,
            workspace=self._workspace,
        )

        # State
        self._tick: int = 0
        self._last_result: SessionResult | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise storage and resume from the last known tick."""
        self._db.connect()
        self._db.ensure_schema()
        self._workspace.ensure_structure()

        # Resume: start one tick past the last recorded tick
        self._tick = self._db.get_last_tick_id() + 1
        log.info("Starting main loop at tick %d", self._tick)

    def stop(self) -> None:
        self._db.close()
        log.info("Main loop stopped at tick %d", self._tick)

    # ── Forever loop ───────────────────────────────────────────────────────

    def run_forever(self) -> None:
        """
        Hot loop — runs until the process is killed.
        Each iteration is one Executive tick (which may spawn a Worker session).
        """
        self.start()
        try:
            while True:
                try:
                    self._run_one_iteration()
                except KeyboardInterrupt:
                    log.info("Interrupted by user")
                    break
                except Exception:
                    log.exception(
                        "Unhandled exception in main loop at tick %d — continuing",
                        self._tick,
                    )
                    # Log crash tick so we don't lose the counter
                    self._safe_log_infra_tick(
                        f"crash: {traceback.format_exc()[-120:]}"
                    )
                    self._tick += 1
        finally:
            self.stop()

    def _run_one_iteration(self) -> None:
        tick = self._tick

        # 1. Health check
        health_issues = self._health.check(tick)

        # 2. Load inbox (gateway hook)
        pending_inbox = self._load_inbox()

        # 3. Executive tick
        actions = self._executive_tick.run(
            current_tick=tick,
            last_result=self._last_result,
            health_issues=health_issues,
            pending_inbox=pending_inbox,
        )
        tick += 1  # Executive tick consumed one tick

        # 4. Execute actions
        task_to_run: Task | None = None

        for action_result in (self._action_executor.execute(a) for a in actions):
            if action_result.error:
                log.warning("Action %r failed: %s", action_result.action.tool, action_result.error)
                continue

            if action_result.task:
                # Only the first assign_task per Executive tick is run immediately.
                # Subsequent ones accumulate next iteration.
                if task_to_run is None:
                    task_to_run = action_result.task
                else:
                    log.warning(
                        "Multiple assign_task in one tick — only running the first; "
                        "subsequent tasks will be re-assigned next tick"
                    )

            if action_result.reflection_requested:
                self._consolidation.request_reflection(
                    action_result.action.params.get("reason", "executive request")
                )

            if action_result.outbox_entry:
                self._deliver_outbox(action_result.outbox_entry)

        # 5. Run Worker session (if a task was assigned)
        if task_to_run:
            session_result = self._run_worker_session(task_to_run, tick)
            self._last_result = session_result
            tick = self._tick  # session updated tick in place
        else:
            self._last_result = None

        # 6. Consolidation
        self._consolidation.maybe_run(tick)

        # 7. Advance tick counter
        self._tick = tick + 1

    # ── Worker session runner ──────────────────────────────────────────────

    def _run_worker_session(self, task: Task, tick_start: int) -> SessionResult:
        """
        Run a full Worker session for the given task.
        Handles retries (up to config.max_task_attempts).
        Updates goal stats in DB after each attempt.
        """
        goal = self._db.get_goal(task.goal_id)
        if not goal:
            log.error("Task assigned for unknown goal %r", task.goal_id)
            # Create a dummy SessionResult
            from ..models import SESSION_STATUS_ERROR_STREAK
            return SessionResult(
                session_id=-1,
                goal_id=task.goal_id,
                task=task,
                status=SESSION_STATUS_ERROR_STREAK,
                summary=f"Goal {task.goal_id} not found",
                tick_start=tick_start,
                tick_end=tick_start,
                action_count=0,
                tokens_used=0,
                transcript_path="",
            )

        previous_summary: str | None = None
        final_result: SessionResult | None = None

        for attempt in range(self._config.max_task_attempts):
            log.info(
                "Starting Worker session for %s (attempt %d/%d)",
                task.goal_id, attempt + 1, self._config.max_task_attempts,
            )

            # Build context
            context_messages = worker_context.build(
                goal=goal,
                task=task,
                workspace=self._workspace,
                db=self._db,
                attempt=attempt,
                previous_summary=previous_summary,
            )

            # Open session in DB and get session_id
            transcript_path = self._workspace.transcript_path(
                goal.workspace_path, session_id=self._tick
            )
            session_id = self._db.open_session(
                goal_id=task.goal_id,
                task=task,
                tick_start=self._tick,
                transcript_path=str(transcript_path),
            )

            # Tick counter is shared by the session (mutated in place)
            tick_counter = [self._tick]

            session = WorkerSession(
                session_id=session_id,
                task=task,
                goal_title=goal.title,
                context_messages=context_messages,
                config=self._config,
                llm=self._llm,
                tools=self._tools,
                db=self._db,
                transcript_path=transcript_path,
                tick_counter=tick_counter,
            )
            result = session.run()

            # Sync tick counter back
            self._tick = tick_counter[0]

            # Persist session result to DB
            self._db.complete_session(session_id, result)

            # Update goal stats
            ticks_used = result.tick_end - result.tick_start
            self._db.update_goal(
                task.goal_id,
                last_worked_tick=self._tick,
                total_ticks=goal.total_ticks + ticks_used,
            )
            # Re-fetch goal with updated totals for next attempt
            goal = self._db.get_goal(task.goal_id) or goal

            final_result = result
            log.info(
                "Session %d for %s: %s (attempt %d)",
                session_id, task.goal_id, result.status, attempt + 1,
            )

            # Decide whether to retry
            from ..models import SESSION_STATUS_DONE, SESSION_STATUS_STUCK
            if result.status == SESSION_STATUS_DONE:
                break  # success — no retry needed
            if result.status == SESSION_STATUS_STUCK:
                break  # stuck — let the Executive decide what to do
            # For max_actions / error_streak / context_overflow → retry

            previous_summary = result.summary

        return final_result  # type: ignore[return-value]

    # ── Gateway hooks (override in subclass for real integration) ──────────

    def _load_inbox(self) -> list[dict]:
        """
        Return a list of unhandled user messages.
        Format: [{"id": str, "text": str, "received_at_tick": int}, ...]

        Default: always empty (no gateway connected).
        Override this in a subclass to wire in the real inbox table/API.
        """
        return []

    def _deliver_outbox(self, entry: dict) -> None:
        """
        Deliver a response to the user.
        entry: {"message_id": str, "text": str}

        Default: just log it.
        Override this in a subclass to write to the outbox table/API.
        """
        log.info(
            "OUTBOX → [%s]: %s",
            entry.get("message_id", "?"),
            entry.get("text", "")[:120],
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _safe_log_infra_tick(self, summary: str) -> None:
        try:
            self._db.log_tick(TickRecord(
                tick_id=self._tick,
                session_id=None,
                goal_id=None,
                actor=ACTOR_INFRA,
                action_type="infra",
                summary=summary[:200],
                outcome=OUTCOME_OK,
            ))
        except Exception:
            pass  # DB might itself be broken; swallow silently
