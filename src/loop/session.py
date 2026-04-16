"""
Worker session — one task execution loop.

A session is bounded by three things:
  1. max_actions_per_session tool calls
  2. max_consecutive_errors consecutive tool errors
  3. context window: once the token count crosses
     config.effective_compress_threshold(), the middle of the conversation
     is compressed via the Summarizer.

Termination reasons and how the main loop interprets them:
  "done"           — task criteria met; log session, mark success
  "stuck"          — Worker is blocked; Executive replans
  "error"          — unrecoverable error state; increment attempts
  "max_actions"    — hit the action limit; session was partial; continue
  "max_errors"     — too many consecutive errors; log and retry
  "context_overflow" — LLM refused due to context length; compress and retry
                       in a new session (the compress happened at threshold,
                       but the LLM still reported overflow — use lower threshold
                       or reduce content)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..characters import summarizer as sum_char
from ..characters import worker as worker_char
from ..config import Config
from ..context import worker as worker_context
from ..db import Database
from ..llm import LLMClient
from ..models import (
    ACTOR_WORKER,
    Goal,
    OUTCOME_ERROR,
    OUTCOME_OK,
    SessionResult,
    Task,
    TickRecord,
)
from ..store import Store
from ..tools import ToolExecutor, WORKER_TOOL_SCHEMAS, format_tool_result_for_llm, format_tool_call_for_transcript
from ..workspace import Workspace
from .turn_guard import TurnGuard

if TYPE_CHECKING:
    from ..audit import AuditLogger

log = logging.getLogger(__name__)


class WorkerSession:
    """
    Run a single task until termination.
    Returns a SessionResult describing the final state.
    """

    def __init__(
        self,
        *,
        config: Config,
        llm: LLMClient,
        db: Database,
        workspace: Workspace,
        store: Store,
        goal: Goal,
        task: Task,
        session_id: int,
        tick_start: int,
        transcript_path: str,
        attempt: int = 0,
        previous_summary: str | None = None,
        audit: "AuditLogger | None" = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._db = db
        self._workspace = workspace
        self._store = store
        self._goal = goal
        self._task = task
        self._session_id = session_id
        self._tick_start = tick_start
        self._transcript_path = transcript_path
        self._attempt = attempt
        self._prev_summary = previous_summary
        self._audit = audit

        # Conversation state
        self._messages: list[dict[str, Any]] = []
        self._transcript_entries: list[str] = []

        # Session counters
        self._action_count = 0
        self._consecutive_errors = 0
        self._tokens_used = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> SessionResult:
        """Run the session loop to completion."""
        initial_cwd = str(self._workspace.goal_work_dir(self._goal.workspace_path))
        self._executor = ToolExecutor(self._config, initial_cwd=initial_cwd)
        self._guard = TurnGuard(llm=self._llm)
        self._init_messages()
        tick_id = self._db.get_last_tick_id() + 1

        while True:
            # Proactive compression before calling the LLM
            self._maybe_compress()

            # Call the LLM — set audit context so the entry carries the right metadata
            self._llm.set_call_context(
                actor=ACTOR_WORKER,
                tick_id=tick_id,
                session_id=self._session_id,
                goal_id=self._goal.goal_id,
            )
            turn = self._guard.call(
                self._messages, WORKER_TOOL_SCHEMAS, temperature=0.3
            )
            self._messages.extend(turn.history_prefix)

            if turn.abort:
                status = "context_overflow" if turn.context_overflow else "stuck"
                return self._end_session(
                    tick_id=tick_id,
                    status=status,
                    summary=turn.abort_reason,
                )

            # Process each tool call
            for tc in turn.tool_calls:
                self._action_count += 1

                # finish() is special — end the session
                if tc.name == "finish":
                    args = tc.arguments
                    status = args.get("status", "done")
                    summary = args.get("summary", "")
                    self._log_tick(
                        tick_id=tick_id,
                        action_type="finish",
                        summary=f"finish({status}): {summary[:120]}",
                        outcome=OUTCOME_OK,
                    )
                    self._write_checkpoint(summary)
                    return self._end_session(
                        tick_id=tick_id,
                        status=status,
                        summary=summary,
                    )

                # Execute the tool
                result = self._executor.execute(tc.name, tc.arguments)
                outcome = OUTCOME_ERROR if result.is_error else OUTCOME_OK

                if result.is_error:
                    self._consecutive_errors += 1
                else:
                    self._consecutive_errors = 0

                # Audit log the tool call (1-month retention)
                if self._audit:
                    try:
                        self._audit.log_tool(
                            actor=ACTOR_WORKER,
                            tick_id=tick_id,
                            session_id=self._session_id,
                            goal_id=self._goal.goal_id,
                            tool_name=tc.name,
                            args=tc.arguments,
                            output=result.output,
                            is_error=result.is_error,
                            truncated=result.truncated,
                        )
                    except Exception as exc:
                        log.warning("audit.log_tool failed: %s", exc)

                # Log to DB + transcript
                self._log_tick(
                    tick_id=tick_id,
                    action_type=tc.name,
                    summary=f"{tc.name}({_args_preview(tc.arguments)}) \u2192 {result.output[:80]}",
                    outcome=outcome,
                )
                self._transcript_entries.append(
                    format_tool_call_for_transcript(
                        tc.name, tc.arguments, result, tick_id
                    )
                )

                # Append tool result to conversation
                self._messages.append(
                    format_tool_result_for_llm(tc.call_id, result)
                )

                tick_id += 1

                # Check termination conditions
                if self._action_count >= self._config.max_actions_per_session:
                    log.info(
                        "Worker hit max_actions_per_session (%d) at tick %d",
                        self._config.max_actions_per_session, tick_id,
                    )
                    return self._end_session(
                        tick_id=tick_id,
                        status="max_actions",
                        summary=(
                            f"Reached action limit ({self._config.max_actions_per_session}). "
                            "Work was partial."
                        ),
                    )

                if self._consecutive_errors >= self._config.max_consecutive_errors:
                    log.warning(
                        "Worker hit max_consecutive_errors (%d) at tick %d",
                        self._config.max_consecutive_errors, tick_id,
                    )
                    return self._end_session(
                        tick_id=tick_id,
                        status="max_errors",
                        summary=(
                            f"{self._consecutive_errors} consecutive errors. "
                            "Ending session to avoid thrashing."
                        ),
                    )

            self._messages.extend(turn.history_suffix)

    # ── Initialisation ─────────────────────────────────────────────────────

    def _init_messages(self) -> None:
        self._messages = [
            {"role": "system", "content": worker_char.SYSTEM_PROMPT},
            *worker_context.build(
                goal=self._goal,
                task=self._task,
                workspace=self._workspace,
                store=self._store,
                db=self._db,
                attempt=self._attempt,
                previous_summary=self._prev_summary,
            ),
        ]

    # ── Compression ────────────────────────────────────────────────────────

    def _maybe_compress(self) -> None:
        """
        If the conversation is approaching the context limit, compress the
        middle messages using the Summarizer.

        The threshold is config.effective_compress_threshold(), which
        defaults to 60 % of context_length_tokens.
        """
        threshold = self._config.effective_compress_threshold()
        if threshold <= 0:
            return

        # Estimate current token count from message content length
        total_chars = sum(len(str(m.get("content") or "")) for m in self._messages)
        estimated_tokens = total_chars // 4  # ~4 chars/token heuristic

        if estimated_tokens < threshold:
            return

        log.info(
            "Worker context at ~%d tokens (threshold %d) \u2014 compressing",
            estimated_tokens, threshold,
        )

        # Keep: system (0), task context (1), last few exchanges
        # Compress: middle messages [2 .. -(KEEP_TAIL*2)]
        keep_tail = 6  # keep last 6 message pairs (assistant + tool result)
        keep_start = 2  # system + context

        if len(self._messages) <= keep_start + keep_tail * 2:
            return  # not enough messages to usefully compress

        to_compress = self._messages[keep_start : -keep_tail]
        tail = self._messages[-keep_tail:]

        try:
            compress_msgs = sum_char.compress_prompt(to_compress)
            self._llm.set_call_context(
                actor=ACTOR_WORKER,
                session_id=self._session_id,
                goal_id=self._goal.goal_id,
            )
            response = self._llm.chat(compress_msgs, temperature=0.2)
            summary_text = response.content or "(no summary)"
        except Exception as exc:
            log.error("Context compression failed: %s", exc)
            return

        compressed_msg: dict[str, Any] = {
            "role": "user",
            "content": (
                f"[Context compressed \u2014 earlier actions summarised]\n\n"
                f"{summary_text}"
            ),
        }

        self._messages = (
            self._messages[:keep_start] + [compressed_msg] + tail
        )
        log.info("Context compressed: %d \u2192 %d messages", len(to_compress), 1)

    # ── Session finalisation ───────────────────────────────────────────────

    def _end_session(
        self, tick_id: int, status: str, summary: str
    ) -> SessionResult:
        # Tear down the persistent shell for this session
        try:
            self._executor.close()
        except Exception as exc:
            log.warning("Failed to close ToolExecutor shell: %s", exc)

        # Write out transcript (to store, not workspace — agent can't tamper)
        transcript_path = self._store.transcript_path(
            self._goal.workspace_path, self._session_id
        )
        if self._transcript_entries:
            try:
                transcript_path.parent.mkdir(parents=True, exist_ok=True)
                with transcript_path.open("w") as f:
                    for line in self._transcript_entries:
                        f.write(line + "\n")
            except Exception as exc:
                log.error("Failed to write transcript: %s", exc)

        return SessionResult(
            session_id=self._session_id,
            goal_id=self._goal.goal_id,
            task=self._task,
            status=status,
            summary=summary,
            tick_start=self._tick_start,
            tick_end=tick_id,
            action_count=self._action_count,
            tokens_used=self._tokens_used,
            transcript_path=self._transcript_path,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _write_checkpoint(self, summary: str) -> None:
        """Write a checkpoint immediately when the worker calls finish()."""
        try:
            prev = self._store.read_checkpoint(self._goal.workspace_path)
            messages = sum_char.checkpoint_prompt(
                task_description=self._task.description,
                session_summary=summary,
                previous_checkpoint=prev,
            )
            self._llm.set_call_context(
                actor=ACTOR_WORKER,
                session_id=self._session_id,
                goal_id=self._goal.goal_id,
            )
            new_content = (self._llm.chat(messages, temperature=0.3).content or "")
            self._store.write_checkpoint(self._goal.workspace_path, new_content)
            log.info("Checkpoint written on finish for %s", self._goal.goal_id)
        except Exception as exc:
            log.warning("Failed to write checkpoint on finish for %s: %s", self._goal.goal_id, exc)

    def _log_tick(
        self, tick_id: int, action_type: str, summary: str, outcome: str
    ) -> None:
        try:
            self._db.log_tick(TickRecord(
                tick_id=tick_id,
                session_id=self._session_id,
                goal_id=self._goal.goal_id,
                actor=ACTOR_WORKER,
                action_type=action_type,
                summary=summary,
                outcome=outcome,
            ))
        except Exception as exc:
            log.error("Failed to log tick %d: %s", tick_id, exc)


def _args_preview(args: dict) -> str:
    parts = []
    for k, v in list(args.items())[:2]:
        v_str = str(v)[:40].replace("\n", "\u21b5")
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)
