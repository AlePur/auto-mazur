"""
Consolidation — hierarchical forgetting.

Runs tick-based maintenance operations that keep the system scalable:

  Every checkpoint_interval ticks:
    Update CHECKPOINT.md for all active goals (using Summarizer)

  Every journal_interval ticks:
    Write a journal entry for each active goal (ticks → Summarizer → .md)

  Every reflection_interval ticks (or on demand):
    Run the Reflector — re-evaluate priorities, distil knowledge

  Every weekly_summary_interval ticks:
    Summarise recent journal entries into a weekly summary doc

  Every archive_interval ticks:
    Archive old ticks from DB → JSONL on disk
    Compress old session transcripts

All LLM calls here are separate from the main agent loop — they use
the Summarizer / Reflector prompts, not the Worker or Executive.
They are invisible to those characters.
"""

from __future__ import annotations

import logging

from .characters import reflector as reflector_char
from .characters import summarizer as sum_char
from .config import Config
from .context import reflector as reflector_context
from .db import Database
from .llm import LLMClient
from .models import (
    ACTOR_INFRA,
    ACTOR_REFLECTOR,
    ACTOR_SUMMARIZER,
    GOAL_STATUS_ACTIVE,
    OUTCOME_ERROR,
    OUTCOME_OK,
    TickRecord,
)
from .workspace import Workspace

log = logging.getLogger(__name__)


class Consolidation:
    def __init__(
        self,
        *,
        config: Config,
        db: Database,
        workspace: Workspace,
        llm: LLMClient,
    ) -> None:
        self._config = config
        self._db = db
        self._workspace = workspace
        self._llm = llm
        self._reflection_requested = False   # set by main loop on Executive request

    def request_reflection(self, reason: str) -> None:
        """Called by the main loop when the Executive requests reflection."""
        self._reflection_requested = True
        self._reflection_reason = reason

    def maybe_run(self, current_tick: int) -> None:
        """
        Called once per main loop iteration, after the Worker session (if any).
        Triggers maintenance operations based on tick schedule.
        """
        cfg = self._config

        # Checkpoint update
        if current_tick % cfg.checkpoint_interval == 0:
            self._update_checkpoints(current_tick)

        # Journal entry
        if current_tick % cfg.journal_interval == 0:
            self._write_journal_entries(current_tick)

        # Weekly summary
        if current_tick % cfg.weekly_summary_interval == 0:
            self._write_weekly_summary(current_tick)

        # Reflection (scheduled or on demand)
        if current_tick % cfg.reflection_interval == 0 or self._reflection_requested:
            reason = getattr(self, "_reflection_reason", "periodic") \
                if self._reflection_requested else "periodic"
            self._run_reflection(current_tick, reason)
            self._reflection_requested = False
            self._reflection_reason = ""

        # Archive old ticks + compress transcripts
        if current_tick % cfg.archive_interval == 0:
            self._archive_old_data(current_tick)

    # ── Individual operations ──────────────────────────────────────────────

    def _update_checkpoints(self, current_tick: int) -> None:
        """
        For each active goal, use the Summarizer to write an updated
        CHECKPOINT.md based on the most recent session summary.
        """
        active_goals = self._db.get_active_goals()
        for goal in active_goals:
            recent = self._db.get_recent_sessions(n=1, goal_id=goal.goal_id)
            if not recent:
                continue
            session = recent[0]
            if not session.get("summary"):
                continue

            prev = self._workspace.read_checkpoint(goal.workspace_path)
            messages = sum_char.checkpoint_prompt(
                task_description=session.get("task_description", ""),
                session_summary=session["summary"],
                previous_checkpoint=prev,
            )
            try:
                response = self._llm.chat(messages, temperature=0.2)
                checkpoint_text = response.content or ""
                if checkpoint_text:
                    self._workspace.write_checkpoint(goal.workspace_path, checkpoint_text)
                    log.debug("Updated checkpoint for %s", goal.goal_id)
            except Exception as exc:
                log.error("Checkpoint update failed for %s: %s", goal.goal_id, exc)

    def _write_journal_entries(self, current_tick: int) -> None:
        """
        For each active goal, summarise recent ticks into a journal entry.
        Uses ticks since the last journal entry.
        """
        journal_interval = self._config.journal_interval
        tick_start = current_tick - journal_interval
        tick_end = current_tick

        active_goals = self._db.get_active_goals()
        for goal in active_goals:
            ticks = self._db.get_ticks_range(tick_start, tick_end)
            goal_ticks = [t for t in ticks if t.goal_id == goal.goal_id]
            if not goal_ticks:
                continue

            messages = sum_char.journal_prompt(goal_ticks, goal.title)
            try:
                response = self._llm.chat(messages, temperature=0.3)
                journal_text = response.content or ""
                if journal_text:
                    self._workspace.append_journal(
                        goal.workspace_path, tick_start, tick_end, journal_text
                    )
                    self._db.log_tick(TickRecord(
                        tick_id=current_tick,
                        session_id=None,
                        goal_id=goal.goal_id,
                        actor=ACTOR_SUMMARIZER,
                        action_type="journal",
                        summary=f"Wrote journal entry ticks {tick_start}-{tick_end}",
                        outcome=OUTCOME_OK,
                    ))
                    log.info("Wrote journal entry for %s", goal.goal_id)
            except Exception as exc:
                log.error("Journal entry failed for %s: %s", goal.goal_id, exc)

    def _write_weekly_summary(self, current_tick: int) -> None:
        """
        Collect recent journal entries across all goals and write a weekly summary.
        """
        all_entries: list[str] = []
        active_goals = self._db.get_active_goals()
        for goal in active_goals:
            entries = self._workspace.read_recent_journals(goal.workspace_path, n=5)
            if entries:
                all_entries.extend(
                    [f"### {goal.title} ({goal.goal_id})\n{e}" for e in entries]
                )

        if not all_entries:
            log.debug("No journal entries for weekly summary at tick %d", current_tick)
            return

        messages = sum_char.weekly_prompt(all_entries)
        try:
            response = self._llm.chat(messages, temperature=0.3)
            weekly_text = response.content or ""
            if weekly_text:
                self._workspace.write_weekly_summary(current_tick, weekly_text)
                self._db.log_tick(TickRecord(
                    tick_id=current_tick,
                    session_id=None,
                    goal_id=None,
                    actor=ACTOR_SUMMARIZER,
                    action_type="weekly_summary",
                    summary=f"Wrote weekly summary at tick {current_tick}",
                    outcome=OUTCOME_OK,
                ))
                log.info("Wrote weekly summary at tick %d", current_tick)
        except Exception as exc:
            log.error("Weekly summary failed: %s", exc)

    def _run_reflection(self, current_tick: int, reason: str) -> None:
        """
        Run the Reflector character.  Applies any updates it produces.
        """
        log.info("Running Reflector at tick %d (reason: %s)", current_tick, reason)

        messages = [
            {"role": "system", "content": reflector_char.SYSTEM_PROMPT},
            *reflector_context.build(
                db=self._db,
                workspace=self._workspace,
                current_tick=current_tick,
                trigger_reason=reason,
            ),
        ]
        try:
            data = self._llm.chat_json(messages, temperature=0.2)
            result = reflector_char.parse_reflector_output(data)
        except Exception as exc:
            log.error("Reflector LLM call failed: %s", exc)
            return

        # Apply priority updates
        for goal_id, new_priority in result.priority_updates:
            try:
                self._db.update_goal(goal_id, priority=new_priority)
                log.info("Reflector: priority of %s → %d", goal_id, new_priority)
            except Exception as exc:
                log.error("Failed to apply priority update for %s: %s", goal_id, exc)

        # Apply goal status changes
        for change in result.goal_status_changes:
            try:
                self._db.update_goal(
                    change.goal_id,
                    status=change.new_status,
                    blocked_reason=change.reason,
                )
                log.info(
                    "Reflector: %s status → %s (%s)",
                    change.goal_id, change.new_status, change.reason
                )
            except Exception as exc:
                log.error("Failed to apply goal status change: %s", exc)

        # Apply knowledge updates
        for ku in result.knowledge_updates:
            try:
                path = self._workspace.write_knowledge(ku.topic, ku.content)
                # One-line summary: first non-empty, non-heading line
                summary_line = next(
                    (
                        line.strip()
                        for line in ku.content.splitlines()
                        if line.strip() and not line.startswith("#")
                    ),
                    "",
                )[:120]
                self._db.upsert_knowledge(
                    topic=ku.topic,
                    file_path=str(path.relative_to(self._workspace.root)),
                    summary=summary_line,
                    tick=current_tick,
                )
                log.info("Reflector: updated knowledge/%s.md", ku.topic)
            except Exception as exc:
                log.error("Failed to write knowledge %s: %s", ku.topic, exc)

        # Write updated PRIORITIES.md
        if result.priorities_md:
            try:
                self._workspace.write_priorities(result.priorities_md)
                log.info("Reflector: updated PRIORITIES.md")
            except Exception as exc:
                log.error("Failed to write PRIORITIES.md: %s", exc)

        # Append observations to REFLECTIONS.md
        if result.observations:
            try:
                header = f"## Reflection at tick {current_tick} ({reason})\n\n"
                self._workspace.append_reflection(header + result.observations)
            except Exception as exc:
                log.error("Failed to append reflection: %s", exc)

        # Log the reflection tick
        obs_preview = result.observations[:100] if result.observations else "no observations"
        self._db.log_tick(TickRecord(
            tick_id=current_tick,
            session_id=None,
            goal_id=None,
            actor=ACTOR_REFLECTOR,
            action_type="reflect",
            summary=(
                f"Reflection ({reason}): "
                f"{len(result.priority_updates)} priority updates, "
                f"{len(result.goal_status_changes)} status changes, "
                f"{len(result.knowledge_updates)} knowledge updates. "
                f"{obs_preview}"
            )[:200],
            outcome=OUTCOME_OK,
        ))

    def _archive_old_data(self, current_tick: int) -> None:
        """
        Archive old ticks from DB → compressed JSONL on disk.
        Compress session transcripts older than archive_interval ticks.
        """
        archive_before = current_tick - self._config.archive_interval
        if archive_before <= 0:
            return

        archive_path = str(
            self._workspace.abs(f"archive/ticks-before-{archive_before}.jsonl")
        )
        try:
            count = self._db.archive_ticks_before(archive_before, archive_path)
            if count:
                self._db.log_tick(TickRecord(
                    tick_id=current_tick,
                    session_id=None,
                    goal_id=None,
                    actor=ACTOR_INFRA,
                    action_type="archive",
                    summary=f"Archived {count} ticks before tick {archive_before}",
                    outcome=OUTCOME_OK,
                ))
        except Exception as exc:
            log.error("Tick archival failed: %s", exc)

        # Compress old session transcripts (all completed .jsonl files)
        all_goals = self._db.get_all_goals()
        for goal in all_goals:
            sessions_dir = self._workspace.abs(goal.workspace_path) / "sessions"
            if not sessions_dir.exists():
                continue
            for jsonl_file in sessions_dir.glob("*.jsonl"):
                try:
                    self._workspace.compress_transcript(jsonl_file)
                except Exception as exc:
                    log.error("Transcript compression failed for %s: %s", jsonl_file, exc)
