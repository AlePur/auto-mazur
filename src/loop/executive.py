"""
Executive tick — one call to the Executive character.

Responsibilities:
  - Assemble the Executive briefing (via context/executive.py)
  - Call the LLM with the Executive's tools
  - Parse the response into ExecutiveAction objects
  - Log the tick to the DB
  - Return the list of actions for the main loop to execute

The Executive always produces at least one action per tick.
If the LLM returns no tool calls (which shouldn't happen with tool_choice="required"),
we log an error tick and return an empty list — the main loop handles this gracefully.
"""

from __future__ import annotations

import logging

from ..characters import executive as exec_char
from ..context import executive as exec_context
from ..config import Config
from ..db import Database
from ..llm import LLMClient
from ..models import (
    ACTOR_EXECUTIVE,
    ExecutiveAction,
    HealthIssue,
    OUTCOME_ERROR,
    OUTCOME_OK,
    SessionResult,
    TickRecord,
)
from ..workspace import Workspace

log = logging.getLogger(__name__)


class ExecutiveTick:
    def __init__(
        self,
        *,
        config: Config,
        llm: LLMClient,
        db: Database,
        workspace: Workspace,
    ) -> None:
        self._config = config
        self._llm = llm
        self._db = db
        self._workspace = workspace

    def run(
        self,
        current_tick: int,
        last_result: SessionResult | None,
        health_issues: list[HealthIssue],
        pending_inbox: list[dict],
    ) -> list[ExecutiveAction]:
        """
        Run one Executive tick.  Logs the tick to DB.
        Returns the list of actions to execute (may be empty on parse failure).
        """
        # Build briefing
        messages = [
            {"role": "system", "content": exec_char.SYSTEM_PROMPT},
            *exec_context.build(
                db=self._db,
                workspace=self._workspace,
                current_tick=current_tick,
                last_result=last_result,
                health_issues=health_issues,
                pending_inbox=pending_inbox,
            ),
        ]

        # Call LLM — require at least one tool call
        try:
            response = self._llm.chat(
                messages=messages,
                tools=exec_char.TOOL_SCHEMAS,
                tool_choice="required",    # Executive must always do something
                temperature=0.5,
            )
        except Exception as exc:
            log.error("Executive LLM call failed at tick %d: %s", current_tick, exc)
            self._log_tick(
                tick_id=current_tick,
                summary=f"LLM call failed: {exc}",
                outcome=OUTCOME_ERROR,
            )
            return []

        # Parse actions
        actions = exec_char.parse_actions(response.tool_calls)

        if not actions:
            log.warning("Executive produced no valid actions at tick %d", current_tick)
            self._log_tick(
                tick_id=current_tick,
                summary="no valid actions produced",
                outcome=OUTCOME_ERROR,
            )
            return []

        # Log tick — one line summary of all actions taken
        action_summary = " + ".join(
            f"{a.tool}({_params_preview(a.params)})" for a in actions
        )
        self._log_tick(
            tick_id=current_tick,
            summary=action_summary[:200],
            outcome=OUTCOME_OK,
        )

        log.info("Executive tick %d: %s", current_tick, action_summary[:120])
        return actions

    # ── Helpers ───────────────────────────────────────────────────────────

    def _log_tick(self, tick_id: int, summary: str, outcome: str) -> None:
        try:
            self._db.log_tick(TickRecord(
                tick_id=tick_id,
                session_id=None,
                goal_id=None,
                actor=ACTOR_EXECUTIVE,
                action_type="decision",
                summary=summary,
                outcome=outcome,
            ))
        except Exception as exc:
            log.error("Failed to log Executive tick %d: %s", tick_id, exc)


def _params_preview(params: dict) -> str:
    """Short preview of action params for logging."""
    parts = []
    for k, v in list(params.items())[:3]:
        v_str = str(v)[:30]
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)
