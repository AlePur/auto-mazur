"""
Worker execution session — the innermost loop.

A session is a multi-turn conversation between the Worker (LLM with tools)
and the real world (shell, files).  It runs until one of these conditions:

  Terminal (clean):
    - Worker calls finish(status="done")    → SESSION_STATUS_DONE
    - Worker calls finish(status="stuck")   → SESSION_STATUS_STUCK
    - Worker calls finish(status="error")   → SESSION_STATUS_ERROR_STREAK

  Terminal (infrastructure limit hit):
    - action_count >= max_actions_per_session → SESSION_STATUS_MAX_ACTIONS
    - consecutive_errors >= max_consecutive_errors → SESSION_STATUS_ERROR_STREAK
    - tokens_used >= context_compress_threshold repeatedly → SESSION_STATUS_CONTEXT_OVERFLOW

Each tool call is ONE tick.  The tick counter is passed in by the caller
and incremented here.  All ticks are logged to the DB.

The conversation is flushed to a JSONL transcript file after each exchange.
Mid-session context compression is handled by the Summarizer when the
conversation grows large.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..characters import worker as worker_char
from ..characters import summarizer as sum_char
from ..config import Config
from ..db import Database
from ..llm import LLMClient
from ..models import (
    ACTOR_WORKER,
    OUTCOME_ERROR,
    OUTCOME_OK,
    SESSION_STATUS_CONTEXT_OVERFLOW,
    SESSION_STATUS_DONE,
    SESSION_STATUS_ERROR_STREAK,
    SESSION_STATUS_MAX_ACTIONS,
    SESSION_STATUS_STUCK,
    SessionResult,
    Task,
    TickRecord,
)
from ..tools import (
    ToolExecutor,
    format_tool_call_for_transcript,
    format_tool_result_for_llm,
)

log = logging.getLogger(__name__)

# How many messages to keep at head and tail during compression
_COMPRESS_HEAD = 2   # system prompt + task context
_COMPRESS_TAIL = 20  # recent tool exchanges


class WorkerSession:
    def __init__(
        self,
        *,
        session_id: int,
        task: Task,
        goal_title: str,
        context_messages: list[dict],   # from context/worker.py build()
        config: Config,
        llm: LLMClient,
        tools: ToolExecutor,
        db: Database,
        transcript_path: Path,
        tick_counter: list[int],         # [current_tick] — mutated in place
    ) -> None:
        self._session_id = session_id
        self._task = task
        self._goal_title = goal_title
        self._config = config
        self._llm = llm
        self._tools = tools
        self._db = db
        self._transcript_path = transcript_path
        self._tick_counter = tick_counter

        # Build the full initial messages list
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": worker_char.SYSTEM_PROMPT},
            *context_messages,
        ]

        self._action_count = 0
        self._consecutive_errors = 0
        self._tokens_used = 0
        self._finish_summary: str = ""
        self._finish_status: str = ""

        # Open transcript file for appending
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self._transcript_file = transcript_path.open("a", encoding="utf-8")

    # ── Main entry point ───────────────────────────────────────────────────

    def run(self) -> SessionResult:
        """Run the session to completion. Returns a SessionResult."""
        tick_start = self._tick_counter[0]
        terminal_status = SESSION_STATUS_MAX_ACTIONS  # default if limits hit

        try:
            terminal_status = self._run_loop()
        except Exception as exc:
            log.exception("Session %d crashed: %s", self._session_id, exc)
            terminal_status = SESSION_STATUS_ERROR_STREAK
        finally:
            self._transcript_file.close()

        tick_end = self._tick_counter[0]
        summary = self._build_summary(terminal_status)

        return SessionResult(
            session_id=self._session_id,
            goal_id=self._task.goal_id,
            task=self._task,
            status=terminal_status,
            summary=summary,
            tick_start=tick_start,
            tick_end=tick_end,
            action_count=self._action_count,
            tokens_used=self._tokens_used,
            transcript_path=str(self._transcript_path),
        )

    # ── Inner loop ─────────────────────────────────────────────────────────

    def _run_loop(self) -> str:
        while True:
            # Check context size before calling LLM
            if self._needs_compression():
                compressed = self._compress_context()
                if not compressed:
                    # Compression failed or we're already at min size
                    return SESSION_STATUS_CONTEXT_OVERFLOW

            # Call the LLM
            try:
                response = self._llm.chat(
                    messages=self._messages,
                    tools=worker_char.WORKER_TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=0.7,
                )
            except Exception as exc:
                log.error("LLM call failed in session %d: %s", self._session_id, exc)
                return SESSION_STATUS_ERROR_STREAK

            self._tokens_used += response.usage.total_tokens

            # No tool calls → model produced text only (unusual but handle it)
            if not response.tool_calls:
                log.warning(
                    "Session %d: LLM returned no tool calls — "
                    "appending as assistant message and continuing",
                    self._session_id,
                )
                self._messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                })
                continue

            # Build the assistant message with tool calls for the conversation
            assistant_msg = self._build_assistant_message(response)
            self._messages.append(assistant_msg)

            # Execute each tool call
            tool_result_messages: list[dict] = []
            for tc in response.tool_calls:
                tick = self._tick_counter[0]

                # ── finish() is intercepted here ─────────────────────────
                if tc.name == "finish":
                    self._finish_summary = tc.arguments.get("summary", "")
                    self._finish_status = tc.arguments.get("status", "done")
                    self._log_tick(
                        tick,
                        action_type="finish",
                        summary=f"finish({self._finish_status}): {self._finish_summary[:100]}",
                        outcome=OUTCOME_OK,
                    )
                    self._tick_counter[0] += 1
                    # Append tool result so conversation is well-formed
                    tool_result_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.call_id,
                        "content": "Session ending.",
                    })
                    # Don't execute any more tool calls this turn
                    self._messages.append(
                        {"role": "tool", "content": json.dumps(tool_result_messages)}
                        if len(tool_result_messages) > 1
                        else tool_result_messages[0]
                    )
                    # Map finish status to session status
                    return {
                        "done": SESSION_STATUS_DONE,
                        "stuck": SESSION_STATUS_STUCK,
                        "error": SESSION_STATUS_ERROR_STREAK,
                    }.get(self._finish_status, SESSION_STATUS_DONE)

                # ── Regular tool execution ────────────────────────────────
                result = self._tools.execute(tc.name, tc.arguments)
                self._action_count += 1

                # Track errors
                if result.is_error:
                    self._consecutive_errors += 1
                else:
                    self._consecutive_errors = 0

                # Log tick
                summary_line = f"{tc.name}({_arg_preview(tc.arguments)}): {result.output[:80]}"
                self._log_tick(
                    tick,
                    action_type=tc.name,
                    summary=summary_line,
                    outcome=OUTCOME_ERROR if result.is_error else OUTCOME_OK,
                )

                # Write to transcript
                self._write_transcript(
                    format_tool_call_for_transcript(tc.name, tc.arguments, result, tick)
                )

                # Append tool result for LLM
                tool_result_messages.append(
                    format_tool_result_for_llm(tc.call_id, result)
                )

                self._tick_counter[0] += 1

            # Add all tool results to conversation
            self._messages.extend(tool_result_messages)

            # ── Guardrail checks ──────────────────────────────────────────
            if self._consecutive_errors >= self._config.max_consecutive_errors:
                log.warning(
                    "Session %d: %d consecutive errors — ending session",
                    self._session_id, self._consecutive_errors,
                )
                return SESSION_STATUS_ERROR_STREAK

            if self._action_count >= self._config.max_actions_per_session:
                log.info(
                    "Session %d: reached max_actions (%d)",
                    self._session_id, self._config.max_actions_per_session,
                )
                return SESSION_STATUS_MAX_ACTIONS

    # ── Context compression ────────────────────────────────────────────────

    def _needs_compression(self) -> bool:
        estimated = LLMClient.estimate_tokens(self._messages)
        return estimated >= self._config.context_compress_threshold_tokens

    def _compress_context(self) -> bool:
        """
        Compress the middle of the conversation using the Summarizer.
        Keeps _COMPRESS_HEAD messages at start and _COMPRESS_TAIL at end.
        Returns True if compression happened, False if not possible.
        """
        if len(self._messages) <= _COMPRESS_HEAD + _COMPRESS_TAIL + 2:
            log.warning("Session %d: context too large but cannot compress further",
                        self._session_id)
            return False

        head = self._messages[:_COMPRESS_HEAD]
        tail = self._messages[-_COMPRESS_TAIL:]
        middle = self._messages[_COMPRESS_HEAD:-_COMPRESS_TAIL]

        log.info(
            "Session %d: compressing %d messages in the middle",
            self._session_id, len(middle),
        )

        compress_messages = sum_char.compress_prompt(middle)
        try:
            response = self._llm.chat(compress_messages, temperature=0.2)
            summary_text = response.content or "(compression produced no output)"
        except Exception as exc:
            log.error("Compression LLM call failed: %s", exc)
            return False

        summary_msg = {
            "role": "user",
            "content": (
                f"[Session history summary — {len(middle)} messages compressed]\n\n"
                f"{summary_text}"
            ),
        }
        self._messages = head + [summary_msg] + tail
        log.info("Session %d: context compressed successfully", self._session_id)
        return True

    # ── Helpers ───────────────────────────────────────────────────────────

    def _log_tick(
        self, tick_id: int, action_type: str, summary: str, outcome: str
    ) -> None:
        try:
            self._db.log_tick(TickRecord(
                tick_id=tick_id,
                session_id=self._session_id,
                goal_id=self._task.goal_id,
                actor=ACTOR_WORKER,
                action_type=action_type,
                summary=summary[:200],
                outcome=outcome,
            ))
        except Exception as exc:
            # DB write failure must not crash the session
            log.error("Failed to log tick %d: %s", tick_id, exc)

    def _write_transcript(self, line: str) -> None:
        try:
            self._transcript_file.write(line + "\n")
            self._transcript_file.flush()
        except Exception as exc:
            log.error("Failed to write transcript: %s", exc)

    def _build_summary(self, status: str) -> str:
        if self._finish_summary:
            return self._finish_summary
        # Auto-generate a minimal summary from status
        return (
            f"Session ended with status '{status}' after "
            f"{self._action_count} tool calls."
        )

    @staticmethod
    def _build_assistant_message(response) -> dict[str, Any]:
        """Build an OpenAI-format assistant message dict with tool_calls."""
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


def _arg_preview(args: dict) -> str:
    """One-line preview of tool arguments for logging."""
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 40:
            v_str = v_str[:40] + "…"
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)
