"""
Build the Reflector's context — what the Reflector sees during a reflection pass.

The Reflector gets a high-level snapshot of the agent's state.
It does NOT see individual tick details (too noisy).
It sees summaries, goals, patterns, and knowledge listings.

The context stays bounded by using journal summaries rather than raw ticks.
"""

from __future__ import annotations

from ..db import Database
from ..workspace import Workspace

_MAX_JOURNAL_ENTRIES = 10   # per goal, max recent journal entries to include
_MAX_WEEKLY_SUMMARIES = 5   # global weekly summaries
_MAX_FAILURES = 20          # recent error-outcome ticks


def build(
    db: Database,
    workspace: Workspace,
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

    # ── All goals ──────────────────────────────────────────────────────────
    all_goals = db.get_all_goals()
    if all_goals:
        lines = ["## All Goals"]
        for g in all_goals:
            last = f"tick {g.last_worked_tick}" if g.last_worked_tick else "never"
            reason = f" ({g.blocked_reason})" if g.blocked_reason else ""
            lines.append(
                f"- `{g.goal_id}` [{g.status}] p{g.priority} "
                f"**{g.title}** — {g.total_ticks} ticks, last: {last}{reason}"
            )
        sections.append("\n".join(lines))

    # ── Recent failures ────────────────────────────────────────────────────
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

    # ── Current PRIORITIES.md ─────────────────────────────────────────────
    priorities = workspace.read_priorities()
    if priorities:
        sections.append(f"## Current PRIORITIES.md\n{priorities}")

    # ── Knowledge index ────────────────────────────────────────────────────
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

    # ── Recent journal entries (across active goals) ───────────────────────
    active_goals = db.get_active_goals()
    journal_sections: list[str] = []
    for g in active_goals[:5]:  # cap at 5 goals to avoid huge context
        entries = workspace.read_recent_journals(g.workspace_path, n=3)
        if entries:
            journal_sections.append(
                f"### {g.title} ({g.goal_id})\n" + "\n\n".join(entries)
            )
    if journal_sections:
        sections.append("## Recent Journal Entries\n" + "\n\n---\n\n".join(journal_sections))

    # ── Weekly summaries ──────────────────────────────────────────────────
    weeklies = workspace.read_weekly_summaries(n=_MAX_WEEKLY_SUMMARIES)
    if weeklies:
        sections.append(
            "## Recent Weekly Summaries\n"
            + "\n\n---\n\n".join(weeklies)
        )

    # ── Recent sessions (cross-goal) ──────────────────────────────────────
    recent_sessions = db.get_recent_sessions(n=10)
    if recent_sessions:
        lines = ["## Recent Sessions"]
        for s in recent_sessions:
            lines.append(
                f"- session {s['session_id']} [{s['goal_id']}] "
                f"`{s.get('status', '?')}`: {s.get('summary', '(no summary)')[:120]}"
            )
        sections.append("\n".join(lines))

    context = "\n\n---\n\n".join(sections)
    return [{"role": "user", "content": context}]
