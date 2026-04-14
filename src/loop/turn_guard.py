"""
TurnGuard — reusable agentic loop safety net.

Provides three protections for any agent that must call tools every turn:

1. NO-TOOL ENFORCEMENT
   When the LLM responds without tool calls, inject a structured error
   message and retry internally (up to max_no_tool_retries times).
   After that, abort with a clean reason.

2. LOOP DETECTION
   Track consecutive calls of the same tool with identical arguments.
   - Soft threshold: inject a warning into the conversation (model self-corrects)
   - Hard threshold: abort the session

3. API AUTO-RETRY
   Transient LLM/API errors are retried with exponential backoff before
   aborting.

Usage pattern:

    guard = TurnGuard(llm=llm)  # one instance per session

    while True:
        result = guard.call(messages, tools)
        messages.extend(result.history_prefix)   # always extend first

        if result.abort:
            return fail(result.abort_reason)

        for tc in result.tool_calls:
            tool_result = execute(tc)
            messages.append(tool_result_msg(tc.call_id, tool_result))

        messages.extend(result.history_suffix)   # loop warnings, if any
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Params that are metadata/tracking, not meaningful tool input.
# Identical tool calls that differ only in these params should still
# be counted as the same call for loop-detection purposes.
_IGNORED_PARAMS: frozenset[str] = frozenset()

# Default patterns that indicate an LLM context-length rejection.
# Passed via TurnPolicy so callers can override or disable.
_DEFAULT_CONTEXT_OVERFLOW_PATTERNS: tuple[str, ...] = (
    "context_length_exceeded",
    "maximum context length",
    "too many tokens",
    "reduce the length",
    "context window",
    "token limit",
)


# ── Default injected messages ─────────────────────────────────────────────────

_DEFAULT_NO_TOOL_MSG = """\
[ERROR] No tool was called in the previous response. You MUST call a tool every turn.

You are operating in an automated pipeline — every response must include a tool call.
- If your objective is complete, call the appropriate completion tool (e.g. `finish`).
- If you need more information, call a query/read tool.
- Otherwise, call the next action tool to make progress.

(Automated message — do not respond conversationally.)"""

_DEFAULT_LOOP_WARNING = (
    "[WARNING] Tool '{tool}' has been called {count} consecutive times with "
    "identical arguments. This is not making progress. "
    "You must use a different tool or supply different arguments."
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TurnPolicy:
    """
    Tunable configuration for TurnGuard.
    Create per-agent variants via TurnPolicy(...) — no subclassing needed.
    """
    # No-tool enforcement
    max_no_tool_retries: int = 2
    # Loop detection
    loop_soft_threshold: int = 3
    loop_hard_threshold: int = 5
    # API error retry
    max_api_retries: int = 3
    api_retry_delays: tuple[float, ...] = (2.0, 4.0, 8.0)
    # Injected message templates (override if needed)
    no_tool_message: str = _DEFAULT_NO_TOOL_MSG
    loop_warning_template: str = _DEFAULT_LOOP_WARNING  # uses {tool} and {count}
    # Context-overflow detection — set to () to disable
    context_overflow_patterns: tuple[str, ...] = _DEFAULT_CONTEXT_OVERFLOW_PATTERNS


@dataclass
class TurnResult:
    """
    Result of one guarded LLM turn.

    Caller pattern:
        messages.extend(result.history_prefix)    # immediately
        if result.abort: ...
        # execute tool_calls, append tool results
        messages.extend(result.history_suffix)    # after tool results
    """
    tool_calls: list                  # validated tool calls to execute (empty on abort)
    history_prefix: list[dict]        # assistant msg(s) + any no-tool retry context
    history_suffix: list[dict]        # loop-detection warnings (append after tool results)
    abort: bool = False
    abort_reason: str = ""
    context_overflow: bool = False    # True → caller should compress and retry


# ── Main class ────────────────────────────────────────────────────────────────

class TurnGuard:
    """
    Stateful safety net for one agent session.
    Create one instance per session (executive tick / worker session run).
    """

    def __init__(self, *, llm: Any, policy: TurnPolicy | None = None) -> None:
        self._llm = llm
        self._policy = policy or TurnPolicy()
        # Per-session loop-detection state
        self._last_tool_name: str = ""
        self._last_tool_params: str = ""   # canonical JSON signature
        self._consecutive_identical_count: int = 0
        # Set by _call_with_api_retry when a context-overflow error is detected
        self._last_error_was_context_overflow: bool = False

    def call(
        self,
        messages: list[dict],
        tools: list,
        *,
        tool_choice: str = "required",
        temperature: float = 0.3,
    ) -> TurnResult:
        """
        Make one guarded LLM turn.

        API retries and no-tool retries are handled internally.
        The caller only sees the final validated result (or an abort).
        """
        policy = self._policy
        working_messages: list[dict] = list(messages)
        history_prefix: list[dict] = []

        for no_tool_attempt in range(policy.max_no_tool_retries + 1):
            # ── API call (with retry) ──────────────────────────────────────
            response = self._call_with_api_retry(
                working_messages, tools, tool_choice, temperature
            )
            if response is None:
                return TurnResult(
                    tool_calls=[],
                    history_prefix=history_prefix,
                    history_suffix=[],
                    abort=True,
                    abort_reason=(
                        "Context window exceeded"
                        if self._last_error_was_context_overflow
                        else "LLM API failed after all retries"
                    ),
                    context_overflow=self._last_error_was_context_overflow,
                )

            assistant_msg = _build_assistant_msg(response)
            history_prefix.append(assistant_msg)
            working_messages.append(assistant_msg)

            # ── No-tool check ─────────────────────────────────────────────
            if not response.tool_calls:
                if no_tool_attempt >= policy.max_no_tool_retries:
                    log.warning(
                        "TurnGuard: no tool called after %d attempt(s) — aborting",
                        no_tool_attempt + 1,
                    )
                    return TurnResult(
                        tool_calls=[],
                        history_prefix=history_prefix,
                        history_suffix=[],
                        abort=True,
                        abort_reason=(
                            f"No tool called after {no_tool_attempt + 1} "
                            f"attempt(s)"
                        ),
                    )

                log.debug(
                    "TurnGuard: no tool in attempt %d — injecting error, retrying",
                    no_tool_attempt + 1,
                )
                error_msg: dict = {"role": "user", "content": policy.no_tool_message}
                history_prefix.append(error_msg)
                working_messages.append(error_msg)
                continue  # retry LLM

            # ── Loop detection ────────────────────────────────────────────
            # Use the first tool call as the representative for loop detection.
            tc = response.tool_calls[0]
            sig = _canonical_signature(tc.arguments)

            if tc.name == self._last_tool_name and sig == self._last_tool_params:
                self._consecutive_identical_count += 1
            else:
                self._consecutive_identical_count = 1

            # Update state AFTER comparison (mirrors Cline's pattern)
            self._last_tool_name = tc.name
            self._last_tool_params = sig

            n = self._consecutive_identical_count

            if n >= policy.loop_hard_threshold:
                log.warning(
                    "TurnGuard: loop hard-escalation — '%s' called %d times identically",
                    tc.name, n,
                )
                return TurnResult(
                    tool_calls=response.tool_calls,
                    history_prefix=history_prefix,
                    history_suffix=[],
                    abort=True,
                    abort_reason=(
                        f"Loop detected: '{tc.name}' called {n} consecutive "
                        f"times with identical arguments"
                    ),
                )

            history_suffix: list[dict] = []
            if n == policy.loop_soft_threshold:
                warning = policy.loop_warning_template.format(tool=tc.name, count=n)
                log.info("TurnGuard: loop soft-warning — '%s' ×%d", tc.name, n)
                history_suffix.append({"role": "user", "content": warning})

            return TurnResult(
                tool_calls=response.tool_calls,
                history_prefix=history_prefix,
                history_suffix=history_suffix,
            )

        # Unreachable — the loop always returns inside
        return TurnResult(
            tool_calls=[],
            history_prefix=history_prefix,
            history_suffix=[],
            abort=True,
            abort_reason="Unexpected TurnGuard loop exit",
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_with_api_retry(
        self,
        messages: list[dict],
        tools: list,
        tool_choice: str,
        temperature: float,
    ) -> Any | None:
        """Call the LLM with exponential-backoff retries on transient errors."""
        policy = self._policy
        self._last_error_was_context_overflow = False
        for attempt in range(policy.max_api_retries + 1):
            try:
                return self._llm.chat(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                )
            except Exception as exc:
                exc_str = str(exc).lower()
                # Context overflow is not a transient error — retrying won't help.
                if policy.context_overflow_patterns and any(
                    p in exc_str for p in policy.context_overflow_patterns
                ):
                    log.warning(
                        "TurnGuard: context overflow detected — aborting immediately: %s",
                        exc,
                    )
                    self._last_error_was_context_overflow = True
                    return None

                if attempt >= policy.max_api_retries:
                    log.error(
                        "TurnGuard: LLM API failed after %d retries: %s",
                        attempt + 1, exc,
                    )
                    return None
                delay = policy.api_retry_delays[
                    min(attempt, len(policy.api_retry_delays) - 1)
                ]
                log.warning(
                    "TurnGuard: LLM API error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, policy.max_api_retries + 1, delay, exc,
                )
                time.sleep(delay)
        return None  # unreachable


# ── Module-level helpers ──────────────────────────────────────────────────────

def _canonical_signature(params: dict | None) -> str:
    """
    Stable JSON signature of tool params, ignoring metadata-only fields.
    Sorted keys so insertion order doesn't affect comparison.
    """
    if not params:
        return "{}"
    keys = sorted(k for k in params if k not in _IGNORED_PARAMS)
    return json.dumps({k: params[k] for k in keys}, sort_keys=True)


def _build_assistant_msg(response: Any) -> dict:
    """Convert an LLM response object into an OpenAI-format assistant message."""
    tool_calls_raw = [
        {
            "id": tc.call_id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.arguments),
            },
        }
        for tc in (response.tool_calls or [])
    ]
    msg: dict = {
        "role": "assistant",
        "content": response.content or None,
    }
    if tool_calls_raw:
        msg["tool_calls"] = tool_calls_raw
    return msg
