"""
Executive tick — one call to the Executive character.

The Executive now operates in two phases per tick:

  1. Query phase — the LLM may call read-only query tools (read_checkpoint,
     read_journal, read_knowledge, search_knowledge, list_sessions,
     read_reflection) to gather more detail before deciding.  The
     infrastructure executes each query tool and feeds the result back.
     The query phase ends when:
       a) the LLM calls one or more action tools, or
       b) max_executive_queries query calls have been made (safety cap).

  2. Decision phase — action tool calls (assign_task, create_goal, etc.)
     are collected from the final LLM response and returned to the main
     loop for execution.

Query tools never mutate state.  Action tools always do.
"""

from __future__ import annotations

import logging
from typing import Any

from ..characters import executive as exec_char
from ..context import executive as exec_context
from ..config import Config
from ..db import Database
from ..llm import LLMClient
from ..models import (
    ACTOR_EXECUTIVE,
    ExecutiveAction,
    HealthIssue,
    OUTCOME_ERROR,
    OUTCOME_OK,
    SessionResult,
    TickRecord,
)
from ..store import Store
from .turn_guard import TurnGuard, TurnPolicy

log = logging.getLogger(__name__)

# Character cap on content returned by query tools to keep context bounded
_QUERY_RESULT_MAX_CHARS = 4_000


class ExecutiveTick:
    def __init__(
        self,
        *,
        config: Config,
        llm: LLMClient,
        db: Database,
        store: Store,
    ) -> None:
        self._config = config
        self._llm = llm
        self._db = db
        self._store = store
        self._guard = TurnGuard(
            llm=self._llm,
            policy=TurnPolicy(max_no_tool_retries=1),
        )

    def run(
        self,
        current_tick: int,
        last_result: SessionResult | None,
        health_issues: list[HealthIssue],
        pending_inbox: list[dict],
    ) -> list[ExecutiveAction]:
        """
        Run one Executive tick (query phase + decision phase).
        Returns the list of actions to execute (may be empty on failure).
        """
        # Build initial briefing
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": exec_char.SYSTEM_PROMPT},
            *exec_context.build(
                db=self._db,
                store=self._store,
                current_tick=current_tick,
                last_result=last_result,
                health_issues=health_issues,
                pending_inbox=pending_inbox,
            ),
        ]

        query_count = 0
        max_queries = self._config.max_executive_queries

        # ── Query / decision loop ──────────────────────────────────────────
        while True:
            self._llm.set_call_context(actor=ACTOR_EXECUTIVE, tick_id=current_tick)
            turn = self._guard.call(
                messages,
                exec_char.ALL_TOOL_SCHEMAS,
                tool_choice="required",
                temperature=0.5,
            )
            messages.extend(turn.history_prefix)

            if turn.abort:
                log.error(
                    "Executive tick %d aborted: %s", current_tick, turn.abort_reason
                )
                self._log_tick(current_tick, turn.abort_reason[:200], OUTCOME_ERROR)
                return []

            # Classify tool calls
            query_calls = [tc for tc in turn.tool_calls
                           if tc.name in exec_char.EXEC_QUERY_TOOLS]
            action_calls = [tc for tc in turn.tool_calls
                            if tc.name not in exec_char.EXEC_QUERY_TOOLS]

            # If any action tools were called, extract and return them
            if action_calls:
                actions = exec_char.parse_actions(action_calls)
                if actions:
                    summary = " + ".join(
                        f"{a.tool}({_params_preview(a.params)})" for a in actions
                    )
                    self._log_tick(current_tick, summary[:200], OUTCOME_OK)
                    log.info("Executive tick %d: %s", current_tick, summary[:120])
                    return actions
                # parse_actions filtered everything out
                self._log_tick(current_tick, "no valid actions produced", OUTCOME_ERROR)
                return []

            # Pure query calls — execute and loop back
            if not query_calls:
                log.warning("Executive returned no recognisable tool calls at tick %d",
                            current_tick)
                self._log_tick(current_tick, "no recognisable tool calls", OUTCOME_ERROR)
                return []

            if query_count >= max_queries:
                log.warning(
                    "Executive hit query cap (%d) at tick %d — forcing decision",
                    max_queries, current_tick,
                )
                # Append a user message forcing a decision
                messages.append({
                    "role": "user",
                    "content": (
                        f"You have used {max_queries} query calls. "
                        "You must now call one or more action tools to make a decision."
                    ),
                })
                # One final LLM call with only action tools
                self._llm.set_call_context(actor=ACTOR_EXECUTIVE, tick_id=current_tick)
                try:
                    response = self._llm.chat(
                        messages=messages,
                        tools=exec_char.ACTION_TOOL_SCHEMAS,
                        tool_choice="required",
                        temperature=0.5,
                    )
                    actions = exec_char.parse_actions(response.tool_calls)
                    summary = " + ".join(
                        f"{a.tool}({_params_preview(a.params)})" for a in actions
                    )
                    self._log_tick(current_tick, summary[:200], OUTCOME_OK)
                    return actions
                except Exception as exc:
                    log.error("Executive forced-decision call failed: %s", exc)
                    self._log_tick(current_tick, f"forced-decision failed: {exc}", OUTCOME_ERROR)
                    return []

            # Execute query tools and append results to conversation
            for tc in query_calls:
                query_count += 1
                result_text = self._execute_query(tc.name, tc.arguments)
                log.debug("Executive query %r \u2192 %d chars", tc.name, len(result_text))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.call_id,
                    "content": result_text,
                })

            messages.extend(turn.history_suffix)

    # ── Query tool execution ───────────────────────────────────────────────

    def _execute_query(self, name: str, args: dict) -> str:
        """Execute one query tool and return its string result (capped)."""
        try:
            result = self._run_query(name, args)
        except Exception as exc:
            return f"[Query error: {exc}]"
        # Cap to avoid flooding context
        if len(result) > _QUERY_RESULT_MAX_CHARS:
            result = result[:_QUERY_RESULT_MAX_CHARS] + "\n...[truncated]"
        return result

    def _run_query(self, name: str, args: dict) -> str:
        match name:
            case "read_checkpoint":
                goal_id = str(args.get("goal_id", ""))
                goal = self._db.get_goal(goal_id)
                if not goal:
                    return f"[Goal {goal_id!r} not found]"
                cp = self._store.read_checkpoint(goal.workspace_path)
                return cp or "(no checkpoint yet)"

            case "read_journal":
                goal_id = str(args.get("goal_id", ""))
                n = min(int(args.get("n", 3)), 10)
                goal = self._db.get_goal(goal_id)
                if not goal:
                    return f"[Goal {goal_id!r} not found]"
                entries = self._db.get_recent_journals(goal_id, n)
                if not entries:
                    return "(no journal entries yet)"
                parts = []
                for e in entries:
                    content = self._store.read_journal_file(e["file_path"]) or "(unreadable)"
                    parts.append(
                        f"### Ticks {e['tick_start']}\u2013{e['tick_end']}\n{content}"
                    )
                return "\n\n---\n\n".join(parts)

            case "read_knowledge":
                topic = str(args.get("topic", ""))
                content = self._store.read_knowledge(topic)
                return content or f"(no knowledge file for topic {topic!r})"

            case "search_knowledge":
                query = str(args.get("query", ""))
                hits = self._db.search_knowledge(query, limit=5)
                if not hits:
                    return "(no knowledge matches)"
                lines = [f"Found {len(hits)} knowledge file(s):"]
                for h in hits:
                    lines.append(f"- **{h['topic']}**: {h['summary']}")
                return "\n".join(lines)

            case "list_sessions":
                goal_id = str(args.get("goal_id", ""))
                n = min(int(args.get("n", 5)), 20)
                sessions = self._db.get_recent_sessions(n=n, goal_id=goal_id)
                if not sessions:
                    return "(no sessions yet)"
                lines = [f"Last {len(sessions)} session(s) for {goal_id}:"]
                for s in sessions:
                    lines.append(
                        f"- session {s['session_id']} "
                        f"[{s.get('status', '?')}] "
                        f"ticks {s['tick_start']}\u2013{s.get('tick_end', '?')}: "
                        f"{(s.get('summary') or '(no summary)')[:120]}"
                    )
                return "\n".join(lines)

            case "read_reflection":
                n = min(int(args.get("n", 2)), 5)
                entries = self._db.get_recent_reflections(n)
                if not entries:
                    return "(no reflections yet)"
                parts = []
                for e in entries:
                    content = self._store.read_reflection_file(e["file_path"]) or "(unreadable)"
                    parts.append(
                        f"### Reflection at tick {e['tick']} ({e['trigger_reason']})\n{content}"
                    )
                return "\n\n---\n\n".join(parts)

            case _:
                return f"[unknown query tool: {name!r}]"

    # ── Helpers ───────────────────────────────────────────────────────────

    def _log_tick(self, tick_id: int, summary: str, outcome: str) -> None:
        try:
            self._db.log_tick(TickRecord(
                tick_id=tick_id,
                session_id=None,
                goal_id=None,
                actor=ACTOR_EXECUTIVE,
                action_type="decision",
                summary=summary,
                outcome=outcome,
            ))
        except Exception as exc:
            log.error("Failed to log Executive tick %d: %s", tick_id, exc)


def _params_preview(params: dict) -> str:
    """Short preview of action params for logging."""
    parts = []
    for k, v in list(params.items())[:3]:
        v_str = str(v)[:30]
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)
