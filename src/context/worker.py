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
  - Any relevant knowledge (keyword matched, capped)
  - Previous attempt summary (if this is a retry)

All content is capped to keep the initial context bounded.
"""

from __future__ import annotations

from ..db import Database
from ..models import Goal, Task
from ..store import Store
from ..workspace import Workspace

# ── Caps ───────────────────────────────────────────────────────────────────
_MAX_KNOWLEDGE_ITEMS    = 5     # knowledge files injected
_MAX_KNOWLEDGE_CHARS    = 800   # chars per knowledge item
_MAX_CHECKPOINT_CHARS   = 3_000 # chars for checkpoint (full detail for worker)


def build(
    goal: Goal,
    task: Task,
    workspace: Workspace,
    store: Store,
    db: Database,
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

    # ── Relevant knowledge (keyword search, capped) ───────────────────────
    keywords = _extract_keywords(task.description, goal.title)
    relevant = _fetch_relevant_knowledge(keywords, db, store)
    if relevant:
        lines = ["## Relevant Notes"]
        for topic, content in relevant:
            truncated = content[:_MAX_KNOWLEDGE_CHARS]
            if len(content) > _MAX_KNOWLEDGE_CHARS:
                truncated += "\n...[truncated]"
            lines.append(f"### {topic}\n{truncated}")
        sections.append("\n\n".join(lines))

    context = "\n\n---\n\n".join(sections)
    return [{"role": "user", "content": context}]


# ── Helpers ───────────────────────────────────────────────────────────────

def _extract_keywords(description: str, title: str) -> str:
    """
    Build a keyword query string from the task description and goal title.
    Simple: take the first 10 significant words.
    """
    import re
    stopwords = {"the", "a", "an", "is", "in", "on", "at", "to", "for",
                 "of", "and", "or", "with", "that", "this", "it", "be", "as"}
    words = re.findall(r"\b[a-zA-Z]{3,}\b", (title + " " + description).lower())
    keywords = [w for w in words if w not in stopwords][:10]
    return " ".join(keywords)


def _fetch_relevant_knowledge(
    query: str,
    db: Database,
    store: Store,
) -> list[tuple[str, str]]:
    """
    Return up to _MAX_KNOWLEDGE_ITEMS (topic, content) pairs.
    Uses FTS5 search from the DB index; reads actual content from disk.
    """
    if not query.strip():
        return []

    try:
        hits = db.search_knowledge(query, limit=_MAX_KNOWLEDGE_ITEMS)
    except Exception:
        # FTS5 search can fail on odd queries — gracefully return nothing
        return []

    results: list[tuple[str, str]] = []
    for hit in hits:
        content = store.read_knowledge(hit["topic"])
        if content:
            results.append((hit["topic"], content))

    return results
