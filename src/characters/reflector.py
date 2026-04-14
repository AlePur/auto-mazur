"""
Reflector character — meta-cognition and priority management.

The Reflector:
  - Has no tools (pure analysis → structured output)
  - Sees journals, all goals, failures, knowledge index
  - Returns structured updates: priority changes, knowledge updates,
    goal status changes, observations
  - Is called by infrastructure — not by the Executive directly
    (the Executive can *request* reflection, but the infrastructure
    decides the exact timing and assembles the context)

Output is JSON. Parsed by parse_reflector_output() into ReflectorResult.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..models import GoalStatusChange, KnowledgeUpdate, ReflectorResult

log = logging.getLogger(__name__)


# ── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the Reflector — the meta-cognitive layer of an autonomous agent.

You receive a snapshot of the agent's current state: its goals, recent \
journal entries, failure patterns, and knowledge base.

Your job is to:
1. Assess whether current priorities make sense
2. Identify patterns in failures or successes
3. Distil important learnings into knowledge files
4. Suggest goal status changes (blocked, paused, abandoned)
5. Rewrite the PRIORITIES.md document if needed
6. Write free-form observations for the agent's reflection log

Be honest and analytical. If a goal is clearly stalled or obsolete, say so.
If the agent keeps hitting the same error, name the pattern.
If something important was learned, write it up as a knowledge update.

Respond ONLY with valid JSON matching this schema:
{
  "priority_updates": [
    {"goal_id": "goal-001", "new_priority": 2}
  ],
  "goal_status_changes": [
    {"goal_id": "goal-003", "new_status": "blocked", "reason": "..."}
  ],
  "knowledge_updates": [
    {
      "topic": "nginx",
      "content": "## Nginx on this machine\\n..."
    }
  ],
  "priorities_md": "# Priorities\\n## Active\\n1. ...",
  "observations": "Free-form reflection notes for the log."
}

Any field may be an empty list / null / empty string if not applicable.
"""


# ── Output parsing ────────────────────────────────────────────────────────

def parse_reflector_output(data: dict[str, Any]) -> ReflectorResult:
    """
    Parse a JSON dict (from chat_json) into a ReflectorResult.
    Gracefully handles missing or malformed fields.
    """
    # priority_updates: list of {goal_id, new_priority}
    priority_updates: list[tuple[str, int]] = []
    for item in data.get("priority_updates") or []:
        try:
            priority_updates.append((str(item["goal_id"]), int(item["new_priority"])))
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("Reflector: invalid priority_update %r — %s", item, exc)

    # goal_status_changes
    goal_status_changes: list[GoalStatusChange] = []
    for item in data.get("goal_status_changes") or []:
        try:
            goal_status_changes.append(GoalStatusChange(
                goal_id=str(item["goal_id"]),
                new_status=str(item["new_status"]),
                reason=str(item.get("reason", "")),
            ))
        except (KeyError, TypeError) as exc:
            log.warning("Reflector: invalid goal_status_change %r — %s", item, exc)

    # knowledge_updates
    knowledge_updates: list[KnowledgeUpdate] = []
    for item in data.get("knowledge_updates") or []:
        try:
            knowledge_updates.append(KnowledgeUpdate(
                topic=str(item["topic"]),
                content=str(item["content"]),
            ))
        except (KeyError, TypeError) as exc:
            log.warning("Reflector: invalid knowledge_update %r — %s", item, exc)

    return ReflectorResult(
        priority_updates=priority_updates,
        goal_status_changes=goal_status_changes,
        knowledge_updates=knowledge_updates,
        priorities_md=data.get("priorities_md") or None,
        observations=str(data.get("observations") or ""),
    )
