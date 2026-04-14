"""
Build the Executive's briefing — what the Executive sees each tick.

Design constraints (for long-running scaling):
  - Active goals: full detail (checkpoint + last session summary)
  - Non-active goals: one-line summary only
  - Done/abandoned goals: just a count
  - Recent decisions: last 10 Executive ticks from DB
  - Inbox: all unhandled messages (expected to be small; gateway archives old ones)

The briefing is built fresh every Executive tick from DB + filesystem state.
Nothing is cached.  Context size stays bounded regardless of tick count.
"""

from __future__ import annotations

from ..db import Database
from ..models import Goal, HealthIssue, SessionResult, GOAL_STATUS_ACTIVE
from ..workspace import Workspace


# Max characters for a checkpoint shown in the Executive briefing.
# If longer, it's truncated.  Full checkpoint is on disk.
_MAX_CHECKPOINT_CHARS = 1500
_MAX_RECENT_TICKS = 10


def build(
    db: Database,
    workspace: Workspace,
    current_tick: int,
    last_result: SessionResult | None,
    health_issues: list[HealthIssue],
    pending_inbox: list[dict],          # [{id, text, received_at_tick}, ...]
) -> list[dict]:
    """
    Return a messages list: [{"role": "user", "content": briefing_text}]
    (system prompt is added by the caller)
    """
    sections: list[str] = []

    # ── Tick counter ───────────────────────────────────────────────────────
    sections.append(f"**Current tick:** {current_tick}")

    # ── Inbox ──────────────────────────────────────────────────────────────
    if pending_inbox:
        lines = [f"## Inbox ({len(pending_inbox)} unread)"]
        for msg in pending_inbox:
            lines.append(f"- [{msg['id']}] {msg['text']}")
        sections.append("\n".join(lines))

    # ── Last session result ────────────────────────────────────────────────
    if last_result:
        goal = db.get_goal(last_result.goal_id)
        goal_title = goal.title if goal else last_result.goal_id
        result_block = (
            f"## Last Worker Session Result\n"
            f"- **Goal:** {goal_title} ({last_result.goal_id})\n"
            f"- **Task:** {last_result.task.description}\n"
            f"- **Outcome:** `{last_result.status}`\n"
            f"- **Summary:** {last_result.summary}\n"
            f"- **Ticks used:** {last_result.tick_end - last_result.tick_start}"
        )
        sections.append(result_block)

    # ── Health issues ──────────────────────────────────────────────────────
    if health_issues:
        lines = ["## ⚠️ Health Alerts"]
        for issue in health_issues:
            lines.append(f"- **{issue.kind}**: {issue.details}")
        sections.append("\n".join(lines))

    # ── Goals ──────────────────────────────────────────────────────────────
    all_goals = db.get_all_goals()
    counts = db.get_goal_counts_by_status()

    active = [g for g in all_goals if g.status == GOAL_STATUS_ACTIVE]
    non_terminal = [
        g for g in all_goals
        if g.status not in ("done", "abandoned", "active")
    ]
    terminal_count = counts.get("done", 0) + counts.get("abandoned", 0)

    # Active goals — full detail
    if active:
        lines = [f"## Active Goals ({len(active)})"]
        for g in active:
            lines.append(_format_active_goal(g, db, workspace))
        sections.append("\n".join(lines))
    else:
        sections.append("## Active Goals\n_(none)_")

    # Paused / blocked goals — one-liner
    if non_terminal:
        lines = [f"## Paused / Blocked Goals ({len(non_terminal)})"]
        for g in non_terminal:
            reason = f" — {g.blocked_reason}" if g.blocked_reason else ""
            lines.append(f"- `{g.goal_id}` [{g.status}] **{g.title}**{reason}")
        sections.append("\n".join(lines))

    # Terminal goals — just a count
    if terminal_count:
        sections.append(f"## Completed / Abandoned Goals\n{terminal_count} total (see DB or archive)")

    # ── Recent Executive decisions ─────────────────────────────────────────
    recent = db.get_recent_ticks(n=_MAX_RECENT_TICKS)
    exec_ticks = [t for t in recent if t.actor == "executive"]
    if exec_ticks:
        lines = ["## Recent Executive Decisions"]
        for t in exec_ticks[-_MAX_RECENT_TICKS:]:
            icon = "✓" if t.outcome == "ok" else "✗"
            lines.append(f"- {icon} tick {t.tick_id}: {t.summary}")
        sections.append("\n".join(lines))

    # ── Reflector observations (if any recent) ─────────────────────────────
    reflector_ticks = [t for t in db.get_recent_ticks(n=50) if t.actor == "reflector"]
    if reflector_ticks:
        last_reflect = reflector_ticks[-1]
        sections.append(
            f"## Last Reflector Pass (tick {last_reflect.tick_id})\n"
            f"{last_reflect.summary}"
        )

    briefing = "\n\n---\n\n".join(sections)
    return [{"role": "user", "content": briefing}]


# ── Helpers ───────────────────────────────────────────────────────────────

def _format_active_goal(g: Goal, db: Database, workspace: Workspace) -> str:
    lines = [
        f"\n### [{g.priority}] {g.title} (`{g.goal_id}`)",
        f"**Description:** {g.description}",
        f"**Total ticks:** {g.total_ticks}  |  **Last worked:** tick {g.last_worked_tick}",
    ]

    # Checkpoint
    checkpoint = workspace.read_checkpoint(g.workspace_path)
    if checkpoint:
        if len(checkpoint) > _MAX_CHECKPOINT_CHARS:
            checkpoint = checkpoint[:_MAX_CHECKPOINT_CHARS] + "\n...[truncated]"
        lines.append(f"**Checkpoint:**\n```\n{checkpoint}\n```")
    else:
        lines.append("**Checkpoint:** _(none yet)_")

    # Last session for this goal
    recent_sessions = db.get_recent_sessions(n=1, goal_id=g.goal_id)
    if recent_sessions:
        s = recent_sessions[0]
        lines.append(
            f"**Last session:** `{s['status']}` — {s.get('summary', '(no summary)')}"
        )

    return "\n".join(lines)
