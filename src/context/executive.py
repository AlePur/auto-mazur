"""
Build the Executive's briefing — what the Executive sees each tick.

Design constraints (for long-running scaling):
  - Active goals: capped at _MAX_ACTIVE_GOALS shown (full detail + recent sessions)
  - Non-active goals: capped at _MAX_NONTERMINAL_GOALS (one-liner each)
  - Done/abandoned goals: just a count
  - Recent decisions: last 10 Executive ticks from DB
  - Knowledge index: topic listing only (use read_knowledge query tool for content)
  - No checkpoints in Executive briefing — checkpoint is Worker-only context

All lists are entry-capped; no unbounded content enters the briefing.
The briefing is built fresh every Executive tick from DB + filesystem state.
Nothing is cached.
"""

from __future__ import annotations

from ..db import Database
from ..models import Goal, HealthIssue, SessionResult, GOAL_STATUS_ACTIVE
from ..store import Store

# ── Caps ───────────────────────────────────────────────────────────────────
_MAX_ACTIVE_GOALS           = 20    # active goals shown in full
_MAX_NONTERMINAL_GOALS      = 20    # paused/blocked goals shown as one-liners
_MAX_RECENT_TICKS           = 10    # recent executive decisions (guaranteed, actor-filtered)
_MAX_SESSIONS_PER_GOAL      = 3     # recent sessions shown per active goal
_MAX_KNOWLEDGE_ENTRIES      = 30    # max knowledge topics listed
_MAX_OUTBOX_ITEMS           = 3     # recent outbox messages shown
_MAX_OUTBOX_CHARS           = 2_000 # truncation per outbox message content


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

    # ── Last session result — always fetched from DB so it persists across ticks
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
            f"_(Use list_sessions or read_journal for full history.)_"
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

    # Active goals — full detail + recent sessions, capped
    if active:
        shown_active = active[:_MAX_ACTIVE_GOALS]
        hidden_active = len(active) - len(shown_active)
        lines = [f"## Active Goals ({len(active)})"]
        for g in shown_active:
            lines.append(_format_active_goal(g, db))
        if hidden_active:
            lines.append(
                f"\n_...{hidden_active} more active goal(s) not shown. "
                "Use list_sessions(goal_id) to inspect them._"
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

    # ── Recent Executive decisions — actor-filtered at DB level ───────────
    exec_ticks = db.get_recent_ticks(n=_MAX_RECENT_TICKS, actor="executive")
    if exec_ticks:
        lines = ["## Recent Executive Decisions"]
        for t in exec_ticks:
            icon = "✓" if t.outcome == "ok" else "✗"
            lines.append(f"- {icon} tick {t.tick_id}: {t.summary}")
        sections.append("\n".join(lines))

    # ── Recent outbox messages ─────────────────────────────────────────────
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

    # ── Knowledge index (topic listing) ───────────────────────────────────
    knowledge_list = db.list_knowledge()
    if knowledge_list:
        shown = knowledge_list[:_MAX_KNOWLEDGE_ENTRIES]
        lines = [f"## Knowledge ({len(knowledge_list)} entries)"]
        for k in shown:
            lines.append(
                f"- `{k['topic']}` (tick {k['updated_at_tick']}): "
                f"{k['summary'] or '(no summary)'}"
            )
        if len(knowledge_list) > _MAX_KNOWLEDGE_ENTRIES:
            lines.append(f"_...{len(knowledge_list) - _MAX_KNOWLEDGE_ENTRIES} more — use search_knowledge_")
        lines.append("_(Use read_knowledge(topic) for full content, search_knowledge(query) to find entries.)_")
        sections.append("\n".join(lines))

    briefing = "\n\n---\n\n".join(sections)
    return [{"role": "user", "content": briefing}]


# ── Helpers ───────────────────────────────────────────────────────────────

def _format_active_goal(g: Goal, db: Database) -> str:
    lines = [
        f"\n### [{g.priority}] {g.title} (`{g.goal_id}`)",
        f"**Description:** {g.description}",
        f"**Total ticks:** {g.total_ticks}  |  **Last worked:** tick {g.last_worked_tick}",
    ]

    # Recent sessions for this goal
    sessions = db.get_recent_sessions(n=_MAX_SESSIONS_PER_GOAL, goal_id=g.goal_id)
    if sessions:
        sess_lines = [f"**Recent sessions** (last {len(sessions)}):"]
        for s in sessions:
            status = s.get("status") or "in-progress"
            summary = (s.get("summary") or "(no summary)")[:100]
            sess_lines.append(
                f"  - session {s['session_id']} [{status}] "
                f"ticks {s.get('tick_start')}–{s.get('tick_end', '?')}: {summary}"
            )
        lines.append("\n".join(sess_lines))
    else:
        lines.append("**Sessions:** _(none yet)_")

    return "\n".join(lines)
