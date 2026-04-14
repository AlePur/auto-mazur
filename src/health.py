"""
Health checker — detects issues in the running agent and returns them
as HealthIssue objects to be injected into the next Executive briefing.

All detection is code-only: no LLM involved.
The Executive sees the issues and decides what to do about them.
"""

from __future__ import annotations

import logging

from .db import Database
from .models import (
    HEALTH_FAILURE_STREAK,
    HEALTH_NEGLECTED_GOAL,
    HEALTH_STUCK_LOOP,
    HealthIssue,
    OUTCOME_ERROR,
)
from .config import Config

log = logging.getLogger(__name__)


class HealthChecker:
    def __init__(self, config: Config, db: Database) -> None:
        self._config = config
        self._db = db

    def check(self, current_tick: int) -> list[HealthIssue]:
        """
        Run all health checks and return a list of detected issues.
        Called once per main loop iteration (before the Executive tick).
        """
        issues: list[HealthIssue] = []
        issues.extend(self._check_stuck_loop())
        issues.extend(self._check_failure_streak())
        issues.extend(self._check_neglected_goals(current_tick))
        return issues

    # ── Individual checks ──────────────────────────────────────────────────

    def _check_stuck_loop(self) -> list[HealthIssue]:
        """
        Detects when the last N action summaries are all identical —
        the agent is doing the same thing repeatedly.
        """
        window = self._config.stuck_detection_window
        summaries = self._db.get_last_n_summaries(window)
        if len(summaries) < window:
            return []

        # All identical → stuck
        if len(set(summaries)) == 1:
            log.warning("Stuck loop detected: repeated action %r", summaries[0][:60])
            return [HealthIssue(
                kind=HEALTH_STUCK_LOOP,
                details=(
                    f"The last {window} actions were identical: "
                    f"{summaries[0][:80]!r}. "
                    "Consider a different approach."
                ),
            )]
        return []

    def _check_failure_streak(self) -> list[HealthIssue]:
        """
        Detects a run of consecutive error outcomes across any actor.
        """
        threshold = self._config.failure_streak_threshold
        outcomes = self._db.get_last_n_outcomes(threshold)
        if len(outcomes) < threshold:
            return []

        if all(o == OUTCOME_ERROR for o in outcomes):
            log.warning("Failure streak detected: %d consecutive errors", threshold)
            return [HealthIssue(
                kind=HEALTH_FAILURE_STREAK,
                details=(
                    f"{threshold} consecutive error outcomes. "
                    "Something may be fundamentally broken. "
                    "Consider stopping and reflecting."
                ),
            )]
        return []

    def _check_neglected_goals(self, current_tick: int) -> list[HealthIssue]:
        """
        Detects active goals that haven't been worked on in a long time.
        Only fires if there are active goals at all — not a sign of neglect
        if the agent has been idle by design.
        """
        threshold = self._config.neglect_threshold_ticks
        cutoff = current_tick - threshold
        if cutoff <= 0:
            return []

        neglected = self._db.get_neglected_goals(threshold_tick=cutoff)
        if not neglected:
            return []

        goal_list = ", ".join(
            f"{g.goal_id} ({g.title}, last: tick {g.last_worked_tick})"
            for g in neglected
        )
        log.warning("Neglected goals: %s", goal_list)
        return [HealthIssue(
            kind=HEALTH_NEGLECTED_GOAL,
            details=(
                f"These active goals have not been worked on in "
                f"{threshold}+ ticks: {goal_list}. "
                "Consider revisiting or pausing them."
            ),
        )]
