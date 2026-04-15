"""
Build the Reflector's context — what the Reflector sees during a reflection pass.

The Reflector gets a high-level snapshot of the agent's state.
It does NOT see individual tick details (too noisy).
It sees summaries, goals, patterns, and knowledge listings.

All lists are entry-capped and content-capped to keep context bounded
regardless of how long the agent has been running.
"""

from __future__ import annotations

from ..db import Database
from ..store import Store

# ── Caps ───────────────────────────────────────────────────────────────────
_MAX_GOALS              = 50    # total goals listed (all statuses)
_MAX_FAILURES           = 20    # recent error-outcome ticks
_MAX_RECENT_SESSIONS    = 10    # recent sessions (cross-goal)
_MAX_ACTIVE_GOALS_JOURNALS = 5  # active goals to pull journal entries for
_MAX_JOURNALS_PER_GOAL  = 3     # journal entries per goal
_MAX_JOURNAL_CHARS      = 2_000 # chars per journal entry
_MAX_WEEKLY_SUMMARIES   = 5     # weekly summaries included
_MAX_WEEKLY_CHARS       = 2_000 # chars per weekly summary
_MAX_PRIORITIES_CHARS   = 3_000 # chars for PRIORITIES.md
_MAX_PREV_REFLECTIONS   = 3     # recent reflections for continuity
_MAX_REFLECTION_CHARS   = 1_500 # chars per previous reflection


def build(
    db: Database,
    store: Store,
    current_tick: int,
    trigger_reason: str,
) -> list[dict]:
    """
    Return a messages list: [{"role": "user", "content": context_text}]
    (system prompt is added by the caller)
    """
    sections: list[str] = []

    sections.append(
        f"**Reflection triggered at tick:** {current_tick}\n"
        f"**Reason:** {trigger_reason}"
    )

    # ── All goals (capped) ─────────────────────────────────────────────────
    all_goals = db.get_all_goals()
    shown_goals = all_goals[:_MAX_GOALS]
    hidden_goals = len(all_goals) - len(shown_goals)
    if shown_goals:
        lines = [f"## All Goals ({len(all_goals)} total)"]
        for g in shown_goals:
            last = f"tick {g.last_worked_tick}" if g.last_worked_tick else "never"
            reason = f" ({g.blocked_reason})" if g.blocked_reason else ""
            lines.append(
                f"- `{g.goal_id}` [{g.status}] p{g.priority} "
                f"**{g.title}** — {g.total_ticks} ticks, last: {last}{reason}"
            )
        if hidden_goals:
            lines.append(f"- _...{hidden_goals} more goals not shown_")
        sections.append("\n".join(lines))

    # ── Recent failures (capped) ───────────────────────────────────────────
    recent_errors = [
        t for t in db.get_recent_ticks(n=200)
        if t.outcome == "error"
    ][-_MAX_FAILURES:]
    if recent_errors:
        lines = [f"## Recent Failures (last {len(recent_errors)})"]
        for t in recent_errors:
            goal = f"[{t.goal_id}]" if t.goal_id else "[no goal]"
            lines.append(f"- tick {t.tick_id} {goal} {t.actor}/{t.action_type}: {t.summary}")
        sections.append("\n".join(lines))

    # ── Current PRIORITIES.md (capped) ────────────────────────────────────
    priorities = store.read_priorities()
    if priorities:
        if len(priorities) > _MAX_PRIORITIES_CHARS:
            priorities = priorities[:_MAX_PRIORITIES_CHARS] + "\n...[truncated]"
        sections.append(f"## Current PRIORITIES.md\n{priorities}")

    # ── Knowledge index (listing only) ────────────────────────────────────
    knowledge_list = db.list_knowledge()
    if knowledge_list:
        lines = ["## Knowledge Files"]
        for k in knowledge_list:
            lines.append(
                f"- `{k['topic']}` (updated tick {k['updated_at_tick']}): "
                f"{k['summary'] or '(no summary)'}"
            )
        sections.append("\n".join(lines))
    else:
        sections.append("## Knowledge Files\n_(none yet)_")

    # ── Recent journal entries (capped per goal, capped per entry) ─────────
    active_goals = db.get_active_goals()
    journal_sections: list[str] = []
    for g in active_goals[:_MAX_ACTIVE_GOALS_JOURNALS]:
        entries = db.get_recent_journals(g.goal_id, _MAX_JOURNALS_PER_GOAL)
        if not entries:
            continue
        goal_parts = []
        for e in entries:
            content = store.read_journal_file(e["file_path"]) or ""
            if len(content) > _MAX_JOURNAL_CHARS:
                content = content[:_MAX_JOURNAL_CHARS] + "\n...[truncated]"
            goal_parts.append(f"_Ticks {e['tick_start']}–{e['tick_end']}_\n{content}")
        journal_sections.append(
            f"### {g.title} ({g.goal_id})\n" + "\n\n".join(goal_parts)
        )
    if journal_sections:
        sections.append(
            "## Recent Journal Entries\n" + "\n\n---\n\n".join(journal_sections)
        )

    # ── Weekly summaries (capped per entry) ───────────────────────────────
    weekly_entries = db.get_recent_weeklies(_MAX_WEEKLY_SUMMARIES)
    if weekly_entries:
        weekly_parts = []
        for w in weekly_entries:
            content = store.read_weekly_file(w["file_path"]) or ""
            if len(content) > _MAX_WEEKLY_CHARS:
                content = content[:_MAX_WEEKLY_CHARS] + "\n...[truncated]"
            weekly_parts.append(f"_Tick {w['tick']}_\n{content}")
        sections.append(
            "## Recent Weekly Summaries\n" + "\n\n---\n\n".join(weekly_parts)
        )

    # ── Recent sessions (cross-goal, summaries only) ───────────────────────
    recent_sessions = db.get_recent_sessions(n=_MAX_RECENT_SESSIONS)
    if recent_sessions:
        lines = ["## Recent Sessions"]
        for s in recent_sessions:
            lines.append(
                f"- session {s['session_id']} [{s['goal_id']}] "
                f"`{s.get('status') or 'orphaned'}`: {(s.get('summary') or '(no summary)')[:120]}"
            )
        sections.append("\n".join(lines))

    # ── Previous reflections (for continuity, capped) ─────────────────────
    prev_reflections = db.get_recent_reflections(_MAX_PREV_REFLECTIONS)
    if prev_reflections:
        lines = [f"## Previous Reflections (last {len(prev_reflections)})"]
        for r in prev_reflections:
            content = store.read_reflection_file(r["file_path"]) or ""
            if len(content) > _MAX_REFLECTION_CHARS:
                content = content[:_MAX_REFLECTION_CHARS] + "\n...[truncated]"
            lines.append(
                f"\n### Tick {r['tick']} ({r['trigger_reason']})\n{content}"
            )
        sections.append("\n".join(lines))

    context = "\n\n---\n\n".join(sections)
    return [{"role": "user", "content": context}]
