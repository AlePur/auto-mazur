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
  Inbox messages stay visible to the Executive every tick until the
  Executive explicitly replies to them via send_user_message(re_message_id=…).
  Sending a reply marks the inbox message answered and writes an outbox row
  with a title + content.  Answered messages remain visible to the Executive
  (for follow-up) for INBOX_ANSWERED_TTL_SECONDS, then are auto-deleted.
"""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from typing import TYPE_CHECKING

from ..config import Config
from ..consolidation import Consolidation
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
from ..workspace import Workspace

if TYPE_CHECKING:
    from ..audit import AuditLogger

log = logging.getLogger(__name__)


class MainLoop:
    def __init__(self, config: Config, audit: "AuditLogger | None" = None) -> None:
        self._config = config
        self._audit = audit

        # Storage
        self._db = Database(config.db_file())
        self._workspace = Workspace(config.workspace_path())

        # LLM (inject audit logger so every call is recorded)
        self._llm = LLMClient(config, audit=audit)

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

        # Gateway HTTP server thread (started lazily in start())
        self._gateway_server = None

        # State
        self._tick: int = 0
        self._last_result: SessionResult | None = None
        self._started: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise storage, resume from the last known tick, and (if
        configured) start the gateway HTTP server in a daemon thread.

        Idempotent: safe to call multiple times (extra calls are no-ops).
        """
        if self._started:
            return
        self._started = True

        self._db.connect()
        self._db.ensure_schema()
        self._workspace.ensure_structure()

        # Resume: start one tick past the last recorded tick
        self._tick = self._db.get_last_tick_id() + 1
        log.info("Starting main loop at tick %d", self._tick)

        # Start the gateway HTTP server (observation + inbox) if enabled
        if self._config.gateway_enabled:
            self._start_gateway()

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
            tick_start_session = self._tick

            session = WorkerSession(
                config=self._config,
                llm=self._llm,
                db=self._db,
                workspace=self._workspace,
                goal=goal,
                task=task,
                session_id=session_id,
                attempt=attempt,
                previous_summary=previous_summary,
                audit=self._audit,
            )
            result = session.run()

            # WorkerSession updates tick via db.get_last_tick_id internally;
            # sync our counter to whatever the session advanced it to.
            self._tick = self._db.get_last_tick_id() + 1

            # Persist session result to DB
            self._db.complete_session(session_id, result)

            # Update goal stats
            ticks_used = result.tick_end - tick_start_session
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

    # ── Inbox / Outbox (DB-backed, used when gateway is enabled) ──────────

    #: Answered inbox messages remain visible to the Executive for this long
    #: before being auto-deleted (3 days).
    INBOX_ANSWERED_TTL_SECONDS: float = 3 * 24 * 3600

    def _load_inbox(self) -> list[dict]:
        """
        Return inbox messages for the Executive's briefing.

        Unanswered messages are included on every tick until the Executive
        explicitly replies to them.  Already-answered messages within the TTL
        window are also included (marked answered=True) so the Executive can
        follow up.  Expired answered messages are pruned here.
        """
        result: list[dict] = []
        try:
            # Prune expired answered messages
            deleted = self._db.delete_expired_inbox(self.INBOX_ANSWERED_TTL_SECONDS)
            if deleted:
                log.debug("Pruned %d expired inbox message(s)", deleted)

            # Unanswered — Executive must respond to these
            for m in self._db.get_pending_inbox():
                result.append({
                    "id": m["msg_id"],
                    "text": m["text"],
                    "received_at": m["received_at"],
                    "answered": False,
                })

            # Answered within TTL — Executive may follow up
            for m in self._db.get_answered_inbox(self.INBOX_ANSWERED_TTL_SECONDS):
                result.append({
                    "id": m["msg_id"],
                    "text": m["text"],
                    "received_at": m["received_at"],
                    "answered": True,
                })
        except Exception as exc:
            log.warning("_load_inbox failed: %s", exc)
        return result

    def _deliver_outbox(self, entry: dict) -> None:
        """
        Persist an Executive message/reply to the outbox table.
        entry: {"title": str, "content": str, "re_message_id": str}

        If re_message_id is set, the corresponding inbox message is marked
        as answered (so it stops appearing in the unanswered section).
        """
        title = entry.get("title", "")
        content = entry.get("content", "")
        re_message_id = entry.get("re_message_id", "")
        log.info("OUTBOX → title=%r re=%r: %s", title, re_message_id or "(none)", content[:120])
        try:
            self._db.add_outbox_entry(
                msg_id=str(uuid.uuid4()),
                reply_to=re_message_id,
                title=title,
                content=content,
                sent_at=time.time(),
            )
            # Mark the inbox message as answered
            if re_message_id:
                self._db.mark_inbox_answered([re_message_id], time.time())
        except Exception as exc:
            log.warning("_deliver_outbox DB write failed: %s", exc)

    # ── Gateway startup ────────────────────────────────────────────────────

    def _start_gateway(self) -> None:
        """Start the GatewayServer in a daemon thread."""
        import threading
        from ..gateway import GatewayServer

        host = self._config.gateway_host
        port = self._config.gateway_port
        self._gateway_server = GatewayServer(
            host=host,
            port=port,
            db=self._db,
            workspace=self._workspace,
            audit=self._audit,
            loop=self,
        )
        thread = threading.Thread(
            target=self._gateway_server.serve_forever,
            daemon=True,
            name="gateway-http",
        )
        thread.start()
        log.info("Gateway HTTP server listening on http://%s:%d", host, port)

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
