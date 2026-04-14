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

import json
import logging
from typing import Any

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
from ..tools import ToolExecutor, WORKER_TOOL_SCHEMAS, format_tool_result_for_llm, format_tool_call_for_transcript
from ..workspace import Workspace

log = logging.getLogger(__name__)

# Error message fragments that indicate an LLM context-length rejection
_CONTEXT_OVERFLOW_PATTERNS = [
    "context_length_exceeded",
    "maximum context length",
    "too many tokens",
    "reduce the length",
    "context window",
    "token limit",
]


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
        goal: Goal,
        task: Task,
        session_id: int,
        attempt: int = 0,
        previous_summary: str | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._db = db
        self._workspace = workspace
        self._goal = goal
        self._task = task
        self._session_id = session_id
        self._attempt = attempt
        self._prev_summary = previous_summary
        self._executor = ToolExecutor(config)

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
        self._init_messages()
        tick_id = self._db.get_last_tick_id() + 1

        while True:
            # Proactive compression before calling the LLM
            self._maybe_compress()

            # Call the LLM
            try:
                response = self._llm.chat(
                    messages=self._messages,
                    tools=WORKER_TOOL_SCHEMAS,
                    tool_choice="required",
                    temperature=0.3,
                )
            except Exception as exc:
                exc_str = str(exc).lower()
                if any(p in exc_str for p in _CONTEXT_OVERFLOW_PATTERNS):
                    log.warning("LLM context overflow at tick %d: %s", tick_id, exc)
                    return self._end_session(
                        tick_id=tick_id,
                        status="context_overflow",
                        summary=(
                            f"Context overflow at action {self._action_count}. "
                            f"Try compressing earlier (lower context_compress_threshold_tokens "
                            f"or reduce max_read_chars)."
                        ),
                    )
                log.error("Worker LLM call failed at tick %d: %s", tick_id, exc)
                return self._end_session(
                    tick_id=tick_id,
                    status="error",
                    summary=f"LLM call failed: {exc}",
                )

            # Handle tool calls
            if not response.tool_calls:
                # No tool call — treat as stuck
                log.warning("Worker produced no tool calls at tick %d", tick_id)
                return self._end_session(
                    tick_id=tick_id,
                    status="stuck",
                    summary="No tool calls returned — model may be confused.",
                )

            # Append assistant message
            self._messages.append(_build_assistant_msg(response))

            # Process each tool call
            for tc in response.tool_calls:
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

                # Log to DB + transcript
                self._log_tick(
                    tick_id=tick_id,
                    action_type=tc.name,
                    summary=f"{tc.name}({_args_preview(tc.arguments)}) → {result.output[:80]}",
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

    # ── Initialisation ─────────────────────────────────────────────────────

    def _init_messages(self) -> None:
        self._messages = [
            {"role": "system", "content": worker_char.SYSTEM_PROMPT},
            *worker_context.build(
                goal=self._goal,
                task=self._task,
                workspace=self._workspace,
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
            "Worker context at ~%d tokens (threshold %d) — compressing",
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
            response = self._llm.chat(compress_msgs, temperature=0.2)
            summary_text = response.content or "(no summary)"
        except Exception as exc:
            log.error("Context compression failed: %s", exc)
            return

        compressed_msg: dict[str, Any] = {
            "role": "user",
            "content": (
                f"[Context compressed — earlier actions summarised]\n\n"
                f"{summary_text}"
            ),
        }

        self._messages = (
            self._messages[:keep_start] + [compressed_msg] + tail
        )
        log.info("Context compressed: %d → %d messages", len(to_compress), 1)

    # ── Session finalisation ───────────────────────────────────────────────

    def _end_session(
        self, tick_id: int, status: str, summary: str
    ) -> SessionResult:
        # Write out transcript
        transcript_path = self._workspace.transcript_path(
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
            tick_end=tick_id,
            action_count=self._action_count,
            tokens_used=self._tokens_used,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

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


# ── Message formatting helpers ─────────────────────────────────────────────

def _build_assistant_msg(response) -> dict[str, Any]:
    tool_calls_raw = []
    for tc in response.tool_calls:
        tool_calls_raw.append({
            "id": tc.call_id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.arguments),
            },
        })
    return {
        "role": "assistant",
        "content": response.content or None,
        "tool_calls": tool_calls_raw,
    }


def _args_preview(args: dict) -> str:
    parts = []
    for k, v in list(args.items())[:2]:
        v_str = str(v)[:40].replace("\n", "↵")
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)
