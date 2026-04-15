"""
Build the Executive's briefing — what the Executive sees each tick.

Design constraints (for long-running scaling):
  - Active goals: capped at _MAX_ACTIVE_GOALS shown (full detail each)
  - Non-active goals: capped at _MAX_NONTERMINAL_GOALS (one-liner each)
  - Done/abandoned goals: just a count
  - Recent decisions: last 10 Executive ticks from DB
  - PRIORITIES.md: included truncated (strategic rationale)
  - Last N reflections: truncated snippet (Executive awareness)
  - No last-session-summary: available on demand via read_checkpoint /
    list_sessions query tools

All lists are entry-capped; no unbounded content enters the briefing.
The briefing is built fresh every Executive tick from DB + filesystem state.
Nothing is cached.
"""

from __future__ import annotations

from ..db import Database
from ..models import Goal, HealthIssue, SessionResult, GOAL_STATUS_ACTIVE
from ..store import Store

# ── Caps ───────────────────────────────────────────────────────────────────
_MAX_ACTIVE_GOALS       = 20    # active goals shown in full
_MAX_NONTERMINAL_GOALS  = 20    # paused/blocked goals shown as one-liners
_MAX_RECENT_TICKS       = 10    # recent executive decisions (guaranteed, actor-filtered)
_MAX_CHECKPOINT_CHARS   = 1_500 # truncation for checkpoint in briefing
_MAX_PRIORITIES_CHARS   = 2_000 # truncation for PRIORITIES.md in briefing
_MAX_REFLECTIONS        = 3     # recent reflections shown as one-liners
_MAX_OUTBOX_ITEMS       = 3     # recent outbox messages shown
_MAX_OUTBOX_CHARS       = 2_000 # truncation per outbox message content


def build(
    db: Database,
    store: Store,
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
    unanswered = [m for m in pending_inbox if not m.get("answered")]
    answered   = [m for m in pending_inbox if m.get("answered")]

    if unanswered:
        lines = [f"## Inbox — {len(unanswered)} message(s) awaiting your reply"]
        for msg in unanswered:
            lines.append(f"- [{msg['id']}] {msg['text']}")
        sections.append("\n".join(lines))

    if answered:
        lines = [
            f"## Recent Messages — {len(answered)} already answered "
            f"(visible for follow-up; use send_user_message with re_message_id to reply again)"
        ]
        for msg in answered:
            lines.append(f"- [{msg['id']}] {msg['text']}")
        sections.append("\n".join(lines))

    # ── Last session result — always fetched from DB so it persists across
    #    ticks (not just the tick immediately following the session).
    recent_sessions = db.get_recent_sessions(n=1)
    if recent_sessions:
        s = recent_sessions[0]
        goal = db.get_goal(s["goal_id"])
        goal_title = goal.title if goal else s["goal_id"]
        result_block = (
            f"## Last Worker Session\n"
            f"- **Goal:** {goal_title} ({s['goal_id']})\n"
            f"- **Task:** {s.get('task_description', '(unknown)')}\n"
            f"- **Outcome:** `{s.get('status') or 'in-progress'}`\n"
            f"- **Summary:** {s.get('summary') or '(no summary yet)'}\n"
            f"- **Ticks:** {s.get('tick_start')}–{s.get('tick_end', '?')}\n"
            f"_(Use list_sessions or read_checkpoint for full detail.)_"
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

    # Active goals — full detail, capped
    if active:
        shown_active = active[:_MAX_ACTIVE_GOALS]
        hidden_active = len(active) - len(shown_active)
        lines = [f"## Active Goals ({len(active)})"]
        for g in shown_active:
            lines.append(_format_active_goal(g, store))
        if hidden_active:
            lines.append(
                f"\n_...{hidden_active} more active goal(s) not shown. "
                "Use read_checkpoint(goal_id) to inspect them._"
            )
        sections.append("\n".join(lines))
    else:
        sections.append("## Active Goals\n_(none)_")

    # Paused / blocked goals — one-liner, capped
    if non_terminal:
        shown_nt = non_terminal[:_MAX_NONTERMINAL_GOALS]
        hidden_nt = len(non_terminal) - len(shown_nt)
        lines = [f"## Paused / Blocked Goals ({len(non_terminal)})"]
        for g in shown_nt:
            reason = f" — {g.blocked_reason}" if g.blocked_reason else ""
            lines.append(f"- `{g.goal_id}` [{g.status}] **{g.title}**{reason}")
        if hidden_nt:
            lines.append(f"- _...{hidden_nt} more not shown_")
        sections.append("\n".join(lines))

    # Terminal goals — just a count
    if terminal_count:
        sections.append(
            f"## Completed / Abandoned Goals\n{terminal_count} total (see DB or archive)"
        )

    # ── Recent Executive decisions — actor-filtered at DB level so we always
    #    get up to _MAX_RECENT_TICKS genuine executive decisions, not mixed.
    exec_ticks = db.get_recent_ticks(n=_MAX_RECENT_TICKS, actor="executive")
    if exec_ticks:
        lines = ["## Recent Executive Decisions"]
        for t in exec_ticks:
            icon = "✓" if t.outcome == "ok" else "✗"
            lines.append(f"- {icon} tick {t.tick_id}: {t.summary}")
        sections.append("\n".join(lines))

    # ── Recent outbox messages (what the Executive last said to the user) ──
    outbox_entries = db.get_recent_outbox(_MAX_OUTBOX_ITEMS)
    if outbox_entries:
        lines = [f"## Recent Outbox Messages (last {len(outbox_entries)})"]
        for entry in outbox_entries:
            title = entry.get("title") or "(no title)"
            content = entry.get("content") or ""
            if len(content) > _MAX_OUTBOX_CHARS:
                content = content[:_MAX_OUTBOX_CHARS] + "\n...[truncated]"
            lines.append(f"### {title}\n{content}")
        sections.append("\n\n".join(lines))

    # ── PRIORITIES.md (strategic rationale) ───────────────────────────────
    priorities = store.read_priorities()
    if priorities:
        if len(priorities) > _MAX_PRIORITIES_CHARS:
            priorities = priorities[:_MAX_PRIORITIES_CHARS] + "\n...[truncated — use read_knowledge to see more]"
        sections.append(f"## Current Priorities\n{priorities}")

    # ── Recent reflections (summaries) ────────────────────────────────────
    recent_reflections = db.get_recent_reflections(_MAX_REFLECTIONS)
    if recent_reflections:
        lines = [f"## Recent Reflections (last {len(recent_reflections)})"]
        for r in recent_reflections:
            lines.append(
                f"- tick {r['tick']} ({r['trigger_reason']}): {r['summary'] or '(no summary)'}"
            )
        lines.append("_(Use read_reflection() for full content.)_")
        sections.append("\n".join(lines))

    briefing = "\n\n---\n\n".join(sections)
    return [{"role": "user", "content": briefing}]


# ── Helpers ───────────────────────────────────────────────────────────────

def _format_active_goal(g: Goal, store: Store) -> str:
    lines = [
        f"\n### [{g.priority}] {g.title} (`{g.goal_id}`)",
        f"**Description:** {g.description}",
        f"**Total ticks:** {g.total_ticks}  |  **Last worked:** tick {g.last_worked_tick}",
    ]

    # Checkpoint (truncated)
    checkpoint = store.read_checkpoint(g.workspace_path)
    if checkpoint:
        if len(checkpoint) > _MAX_CHECKPOINT_CHARS:
            checkpoint = checkpoint[:_MAX_CHECKPOINT_CHARS] + "\n...[truncated — use read_checkpoint() for full]"
        lines.append(f"**Checkpoint:**\n```\n{checkpoint}\n```")
    else:
        lines.append("**Checkpoint:** _(none yet)_")

    return "\n".join(lines)
