"""
Build the Worker's task context — what the Worker sees at the start of a session.

The Worker sees NOTHING about:
  - Other goals or priorities
  - The user or inbox
  - Tick counts or system architecture
  - Health issues

It only sees what it needs to do the work:
  - The task description and success criteria
  - Where to find its working directory
  - The checkpoint from last time (capped)
  - Previous attempt summary (if this is a retry)

The Worker has a search_knowledge() tool to look up relevant knowledge
on demand — nothing is pre-injected into the initial context.

All content is capped to keep the initial context bounded.
"""

from __future__ import annotations

from ..models import Goal, Task
from ..store import Store
from ..workspace import Workspace

# ── Caps ───────────────────────────────────────────────────────────────────
_MAX_CHECKPOINT_CHARS   = 3_000 # chars for checkpoint (full detail for worker)


def build(
    goal: Goal,
    task: Task,
    workspace: Workspace,
    store: Store,
    attempt: int = 0,
    previous_summary: str | None = None,
) -> list[dict]:
    """
    Return a messages list: [{"role": "user", "content": context_text}]
    (system prompt is added by the caller)
    """
    sections: list[str] = []

    # ── Task ───────────────────────────────────────────────────────────────
    sections.append(
        f"## Task\n{task.description}\n\n"
        f"## Success Criteria\n{task.criteria}"
    )

    # ── Working directory ──────────────────────────────────────────────────
    goal_abs = workspace.root / goal.workspace_path
    sections.append(
        f"## Working Directory\n"
        f"`{goal_abs}`\n\n"
        f"Your goal's workspace is at this path. You can read and write "
        f"files here freely. Use shell() to explore."
    )

    # ── Checkpoint (where we left off) — capped ───────────────────────────
    checkpoint = store.read_checkpoint(goal.workspace_path)
    if checkpoint:
        if len(checkpoint) > _MAX_CHECKPOINT_CHARS:
            checkpoint = checkpoint[:_MAX_CHECKPOINT_CHARS] + "\n...[truncated]"
        sections.append(f"## Where You Left Off\n{checkpoint}")

    # ── Retry context ──────────────────────────────────────────────────────
    if attempt > 0 and previous_summary:
        sections.append(
            f"## Previous Attempt (attempt {attempt})\n"
            f"{previous_summary}\n\n"
            "Consider a different approach this time."
        )

    context = "\n\n---\n\n".join(sections)
    return [{"role": "user", "content": context}]
