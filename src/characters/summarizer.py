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

from ..models import TickRecord


# ── Journal entry — summarise recent ticks ───────────────────────────────

_JOURNAL_SYSTEM = """\
You are summarising the recent actions of an autonomous agent for its journal.

Write a concise journal entry in markdown. Focus on:
- What was accomplished
- What failed and why (if anything)
- Key decisions made
- Important things learned

Be specific: mention file paths, error messages, command names when relevant.
Be concise: aim for 150-300 words. No fluff.
"""


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

_COMPRESS_SYSTEM = """\
You are summarising the middle portion of an autonomous agent's work session.

The agent is working on a task. You are given the conversation history from
the middle of the session (not the beginning or end).

Produce a concise summary that the agent can use to restore context. Include:
- What the agent was trying to do at this point
- Commands run and their results (especially any errors)
- Files read or written
- Current state of the work
- Any important values, paths, or findings

Be specific and factual. This summary replaces the raw messages in the
agent's context window, so include everything it would need to continue.
"""


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

_WEEKLY_SYSTEM = """\
You are writing a weekly summary for an autonomous agent.

You are given a set of recent journal entries. Synthesise them into a
weekly summary that captures:
- Major accomplishments across all goals
- Persistent problems or patterns
- The current state of each active goal
- Anything that should be remembered going forward

Aim for 200-400 words in markdown format.
"""


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

_CHECKPOINT_SYSTEM = """\
You are writing a CHECKPOINT.md for an autonomous agent goal.

The checkpoint is read at the start of every work session on this goal.
It should tell the agent exactly where it left off and what to do next.

Format:
## Where I left off
(1-2 sentences)

## Next steps
1. ...
2. ...
3. ...

## Gotchas / notes
(any non-obvious things to remember)

Be specific and concrete.
"""


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
