"""
Summarizer character — utility for compressing information.

The Summarizer is stateless: each call is independent.
It has no tools, just text in → text out.

Used in three contexts:
  1. journal_prompt()     — summarise N ticks into a journal entry
  2. compress_prompt()    — compress the middle of a conversation transcript
  3. weekly_prompt()      — summarise N journal entries into a weekly summary
  4. session_prompt()     — summarise a completed Worker session transcript

Each returns a (system, user) tuple of message dicts ready to pass to
LLMClient.chat_json() or LLMClient.chat().
"""

from __future__ import annotations

from pathlib import Path

from ..models import TickRecord

_SOULS = Path(__file__).parent.parent / "souls"

# ── Journal entry — summarise recent ticks ───────────────────────────────

_JOURNAL_SYSTEM = (_SOULS / "summarizer-journal.md").read_text()


def journal_prompt(ticks: list[TickRecord], goal_title: str) -> list[dict]:
    lines = []
    for t in ticks:
        status = "✓" if t.outcome == "ok" else "✗"
        lines.append(f"{status} [{t.actor}] {t.action_type}: {t.summary}")
    tick_block = "\n".join(lines) or "(no ticks)"

    return [
        {"role": "system", "content": _JOURNAL_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Goal: {goal_title}\n\n"
                f"Recent actions (oldest → newest):\n{tick_block}\n\n"
                "Write the journal entry."
            ),
        },
    ]


# ── Context window compression ────────────────────────────────────────────

_COMPRESS_SYSTEM = (_SOULS / "summarizer-compress.md").read_text()


def compress_prompt(messages: list[dict]) -> list[dict]:
    """
    Build messages for compressing the middle of a Worker conversation.
    `messages` is the slice to compress (not system or task context).
    """
    content_parts = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if m.get("role") == "assistant" and m.get("tool_calls"):
            # Format tool calls compactly
            calls = m["tool_calls"]
            if isinstance(calls, list):
                tc_lines = []
                for tc in calls:
                    if isinstance(tc, dict):
                        name = tc.get("function", {}).get("name", "?")
                        tc_lines.append(f"  → {name}(...)")
                content = "[tool calls]\n" + "\n".join(tc_lines)
        content_parts.append(f"[{role}]: {content[:1000]}")

    raw = "\n\n".join(content_parts)
    return [
        {"role": "system", "content": _COMPRESS_SYSTEM},
        {
            "role": "user",
            "content": f"Conversation to summarise:\n\n{raw}\n\nWrite the summary.",
        },
    ]


# ── Weekly summary — summarise journal entries ────────────────────────────

_WEEKLY_SYSTEM = (_SOULS / "summarizer-weekly.md").read_text()


def weekly_prompt(journal_entries: list[str]) -> list[dict]:
    combined = "\n\n---\n\n".join(journal_entries) or "(no journal entries)"
    return [
        {"role": "system", "content": _WEEKLY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Journal entries to summarise:\n\n{combined}\n\n"
                "Write the weekly summary."
            ),
        },
    ]


# ── Checkpoint update — summarise current goal state ─────────────────────

_CHECKPOINT_SYSTEM = (_SOULS / "summarizer-checkpoint.md").read_text()


def checkpoint_prompt(
    task_description: str,
    session_summary: str,
    previous_checkpoint: str | None,
) -> list[dict]:
    prev = previous_checkpoint or "(none)"
    return [
        {"role": "system", "content": _CHECKPOINT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Task just completed: {task_description}\n"
                f"Session summary: {session_summary}\n"
                f"Previous checkpoint:\n{prev}\n\n"
                "Write the updated CHECKPOINT.md."
            ),
        },
    ]
