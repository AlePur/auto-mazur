"""
Consolidation — background memory maintenance.

Runs after every Worker session completes (or on demand from the Executive).

Tasks:
  1. Checkpoint — update CHECKPOINT.md for the goal (Worker-only, every N ticks).
  2. Journal     — write a journal entry for a goal when it exceeds the
                   activity threshold or is explicitly requested.
  3. Weekly      — summarise all goal journals into a weekly digest when
                   the global tick counter crosses the weekly interval.
  4. Archive     — prune old ticks to keep the DB lean.

Reflection and PRIORITIES.md have been removed; knowledge management is now
the Executive's direct responsibility via write_knowledge/forget_knowledge.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .characters.summarizer import (
    checkpoint_prompt,
    journal_prompt,
    weekly_prompt,
)
from .config import Config
from .db import Database
from .llm import LLMClient
from .models import (
    ACTOR_SUMMARIZER,
    OUTCOME_ERROR,
    OUTCOME_OK,
    SessionResult,
    TickRecord,
)
from .store import Store

log = logging.getLogger(__name__)


class Consolidation:
    """
    Handles all post-session memory maintenance.

    Call ``run()`` after each Worker session completes.
    Call ``journal_goal()`` directly when the Executive requests journaling.
    """

    def __init__(
        self,
        *,
        config: Config,
        llm: LLMClient,
        db: Database,
        store: Store,
    ) -> None:
        self._config = config
        self._llm = llm
        self._db = db
        self._store = store

    def run(self, result: SessionResult, current_tick: int) -> None:
        """Run all consolidation tasks after a Worker session."""
        goal = self._db.get_goal(result.goal_id)
        if not goal:
            log.warning("Consolidation: goal %r not found", result.goal_id)
            return

        # 1. Checkpoint (every N ticks for this goal)
        self._maybe_write_checkpoint(result, goal.workspace_path, current_tick)

        # 2. Goal journal (auto — based on activity threshold)
        self._maybe_journal_goal(result.goal_id, goal.workspace_path, goal.title, current_tick)

        # 3. Weekly digest (global — based on global tick counter)
        self._maybe_write_weekly(current_tick)

        # 4. Tick archive (global)
        self._maybe_archive_ticks(current_tick)

    def journal_goal(self, goal_id: str, current_tick: int) -> bool:
        """
        Explicitly write a journal entry for a goal (called by Executive's
        request_journaling action).  Returns True if successful.
        """
        goal = self._db.get_goal(goal_id)
        if not goal:
            log.error("journal_goal: goal %r not found", goal_id)
            return False
        return self._write_journal_for_goal(
            goal_id=goal_id,
            workspace_path=goal.workspace_path,
            goal_title=goal.title,
            current_tick=current_tick,
            force=True,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _maybe_write_checkpoint(
        self,
        result: SessionResult,
        workspace_path: str,
        current_tick: int,
    ) -> None:
        interval = self._config.checkpoint_interval
        if current_tick % interval != 0 and interval > 0:
            return
        try:
            prev = self._store.read_checkpoint(workspace_path)
            messages = checkpoint_prompt(
                task_description=result.task.description,
                session_summary=result.summary,
                previous_checkpoint=prev,
            )
            new_content = (self._llm.chat(messages, temperature=0.3).content or "")
            self._store.write_checkpoint(workspace_path, new_content)
            log.info("Checkpoint written for %s at tick %d", result.goal_id, current_tick)
            self._db.log_tick(TickRecord(
                tick_id=current_tick,
                session_id=result.session_id,
                goal_id=result.goal_id,
                actor=ACTOR_SUMMARIZER,
                action_type="checkpoint",
                summary=f"Checkpoint updated for {result.goal_id}",
                outcome=OUTCOME_OK,
            ))
        except Exception as exc:
            log.error("Checkpoint failed for %s: %s", result.goal_id, exc)
            self._db.log_tick(TickRecord(
                tick_id=current_tick,
                session_id=result.session_id,
                goal_id=result.goal_id,
                actor=ACTOR_SUMMARIZER,
                action_type="checkpoint",
                summary=f"Checkpoint FAILED for {result.goal_id}: {exc}",
                outcome=OUTCOME_ERROR,
            ))

    def _maybe_journal_goal(
        self,
        goal_id: str,
        workspace_path: str,
        goal_title: str,
        current_tick: int,
    ) -> None:
        """Auto-journal if the goal has accumulated enough un-journaled ticks."""
        threshold = self._config.journal_activity_threshold
        if threshold <= 0:
            return

        # Find the tick of the last journal entry for this goal
        recent_journals = self._db.get_recent_journals(goal_id, 1)
        last_journal_tick = (
            recent_journals[-1]["tick_end"] if recent_journals else 0
        )

        # Count goal ticks since last journal
        goal_ticks = self._db.get_recent_ticks(n=threshold + 1, goal_id=goal_id)
        new_ticks = [t for t in goal_ticks if t.tick_id > last_journal_tick]

        if len(new_ticks) < threshold:
            return

        self._write_journal_for_goal(
            goal_id=goal_id,
            workspace_path=workspace_path,
            goal_title=goal_title,
            current_tick=current_tick,
            force=False,
        )

    def _write_journal_for_goal(
        self,
        goal_id: str,
        workspace_path: str,
        goal_title: str,
        current_tick: int,
        force: bool = False,
    ) -> bool:
        """Write a journal entry for the goal. Returns True on success."""
        # Gather last 50 ticks for the goal
        ticks = self._db.get_recent_ticks(n=100, goal_id=goal_id)
        if not ticks and not force:
            return False

        # Determine tick range covered
        tick_start = ticks[0].tick_id if ticks else current_tick
        tick_end = current_tick

        # Gather last 5 session results for the goal
        sessions = self._db.get_recent_sessions(n=5, goal_id=goal_id)

        # Gather last 5 previous journal entry contents
        recent_journal_meta = self._db.get_recent_journals(goal_id, 5)
        previous_journals: list[str] = []
        for jmeta in recent_journal_meta:
            content = self._store.read_journal_file(jmeta["file_path"])
            if content:
                previous_journals.append(content)

        try:
            messages = journal_prompt(
                ticks=ticks,
                goal_title=goal_title,
                sessions=sessions if sessions else None,
                previous_journals=previous_journals if previous_journals else None,
            )
            entry_content = (self._llm.chat(messages, temperature=0.3).content or "")

            # Persist the file
            journal_path = self._store.append_journal(
                workspace_path=workspace_path,
                tick_start=tick_start,
                tick_end=tick_end,
                content=entry_content,
            )

            # Store-relative path for the DB index
            rel_path = str(journal_path.relative_to(self._store.root))

            # Extract summary (first non-empty line)
            summary_line = next(
                (line.strip() for line in entry_content.splitlines() if line.strip()),
                "(journal entry)"
            )[:120]

            self._db.upsert_journal(
                goal_id=goal_id,
                tick_start=tick_start,
                tick_end=tick_end,
                file_path=rel_path,
                summary=summary_line,
            )

            log.info(
                "Journal entry written for %s (ticks %d–%d)", goal_id, tick_start, tick_end
            )
            self._db.log_tick(TickRecord(
                tick_id=current_tick,
                session_id=None,
                goal_id=goal_id,
                actor=ACTOR_SUMMARIZER,
                action_type="journal",
                summary=f"Journal entry for {goal_id}: {summary_line[:80]}",
                outcome=OUTCOME_OK,
            ))
            return True

        except Exception as exc:
            log.error("Journal failed for %s: %s", goal_id, exc)
            self._db.log_tick(TickRecord(
                tick_id=current_tick,
                session_id=None,
                goal_id=goal_id,
                actor=ACTOR_SUMMARIZER,
                action_type="journal",
                summary=f"Journal FAILED for {goal_id}: {exc}",
                outcome=OUTCOME_ERROR,
            ))
            return False

    def _maybe_write_weekly(self, current_tick: int) -> None:
        interval = self._config.weekly_summary_interval
        if interval <= 0 or current_tick % interval != 0:
            return
        try:
            all_goals = self._db.get_all_goals()
            journal_texts: list[str] = []
            for g in all_goals:
                for jmeta in self._db.list_journals_for_goal(g.goal_id):
                    content = self._store.read_journal_file(jmeta["file_path"])
                    if content:
                        journal_texts.append(
                            f"### {g.title} ({g.goal_id})\n{content}"
                        )

            messages = weekly_prompt(journal_texts)
            weekly_content = (self._llm.chat(messages, temperature=0.3).content or "")
            path = self._store.write_weekly_summary(current_tick, weekly_content)
            rel = str(path.relative_to(self._store.root))
            summary_line = next(
                (l.strip() for l in weekly_content.splitlines() if l.strip()),
                "(weekly summary)"
            )[:120]
            self._db.upsert_weekly(current_tick, rel, summary_line)
            log.info("Weekly summary written at tick %d", current_tick)
        except Exception as exc:
            log.error("Weekly summary failed at tick %d: %s", current_tick, exc)

    def _maybe_archive_ticks(self, current_tick: int) -> None:
        interval = self._config.archive_interval
        if interval <= 0 or current_tick % interval != 0:
            return
        try:
            archive_path = str(
                self._store.root / "archive" / f"ticks-before-{current_tick}.jsonl"
            )
            archived = self._db.archive_ticks_before(
                current_tick - interval, archive_path
            )
            if archived:
                log.info("Archived %d ticks (before %d)", archived, current_tick - interval)
        except Exception as exc:
            log.error("Tick archive failed at tick %d: %s", current_tick, exc)
