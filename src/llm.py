"""
LLM client — thin wrapper around the vLLM OpenAI-compatible chat completions API.

Responsibilities:
  - Retry with exponential backoff on transient errors
  - Normalise responses into our internal LLMResponse type
  - Rough token counting for context-budget tracking
  - No prompt logic lives here; that belongs to characters/

Uses httpx directly (no openai SDK) so there is no third-party OpenAI dependency.
Set config.api_base and config.model to point at the vLLM server on Tailscale.

Thinking/reasoning (Gemma 4):
  - Every request includes chat_template_kwargs: {enable_thinking: true}.
  - The model's reasoning chain is captured in LLMResponse.thinking and written
    to the audit log (if an AuditLogger is injected).  It is NEVER forwarded
    back into any agent message list — doing so would pollute the context window.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

import httpx

from .config import Config
from .models import LLMResponse, ToolCall, Usage

if TYPE_CHECKING:
    from .audit import AuditLogger

log = logging.getLogger(__name__)

# Rough token estimate: 1 token ≈ 4 chars for most LLMs.
# Used for budget tracking when the API doesn't return usage.
_CHARS_PER_TOKEN = 4

# HTTP status codes that are retried (rate-limit and server errors).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# Context carried through one LLM call for audit annotation.
# Set via LLMClient.set_call_context() before each call.
_CallContext = dict  # {"actor": str, "tick_id": int|None, "session_id": int|None, "goal_id": str|None}


class LLMClient:
    def __init__(self, config: Config, audit: "AuditLogger | None" = None) -> None:
        self._cfg = config
        self._audit = audit
        self._http = httpx.Client(
            base_url=config.api_base,
            headers={"Content-Type": "application/json"},
            # Generous timeout: vLLM may take a while for long reasoning chains.
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
        )
        # Current call context — set by callers before each LLM call so audit
        # entries carry the right actor/tick/session/goal metadata.
        self._call_ctx: _CallContext = {
            "actor": "unknown",
            "tick_id": None,
            "session_id": None,
            "goal_id": None,
        }

    def set_audit(self, audit: "AuditLogger") -> None:
        """Attach (or replace) the audit logger after construction."""
        self._audit = audit

    def set_call_context(
        self,
        actor: str,
        tick_id: int | None = None,
        session_id: int | None = None,
        goal_id: str | None = None,
    ) -> None:
        """
        Set metadata that will be attached to the next audit log entry.
        Call this immediately before chat() / chat_json() so the entry
        carries the correct actor and tick information.
        """
        self._call_ctx = {
            "actor": actor,
            "tick_id": tick_id,
            "session_id": session_id,
            "goal_id": goal_id,
        }

    # ── Main interface ─────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        - Retries up to config.max_retries on rate-limit / server errors.
        - Thinking mode is always enabled (chat_template_kwargs).
        - tool_choice: "auto" | "none" | {"type": "function", "function": {"name": "..."}}
          Defaults to "auto" when tools are provided, otherwise omitted.
        """
        body: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "temperature": temperature,
            # Always enable Gemma 4 structured thinking.  The reasoning chain
            # improves answer quality; we discard the tokens, keep only content.
            "chat_template_kwargs": {"enable_thinking": True},
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"

        raw = self._call_with_retry(body)
        resp = self._parse_response(raw)
        self._audit_llm(resp)
        return resp

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """
        Like chat() but instructs the model to return valid JSON and
        parses the response.  Used for structured output (Reflector, etc.)
        """
        body: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": True},
        }
        raw = self._call_with_retry(body)
        resp = self._parse_response(raw)
        self._audit_llm(resp)
        text = resp.content or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning("chat_json: invalid JSON from model — %s\nRaw: %s", exc, text[:500])
            return {}

    # ── Context budget helper ──────────────────────────────────────────────

    @staticmethod
    def estimate_tokens(messages: list[dict[str, Any]]) -> int:
        """
        Rough token count for a list of messages.
        Accurate enough for budget tracking; not a substitute for a tokeniser.
        """
        total_chars = sum(
            len(str(m.get("content") or "")) + len(str(m.get("role") or ""))
            for m in messages
        )
        return total_chars // _CHARS_PER_TOKEN

    # ── Internal ───────────────────────────────────────────────────────────

    def _call_with_retry(self, body: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self._cfg.max_retries):
            try:
                response = self._http.post("/chat/completions", json=body)
                if response.status_code in _RETRYABLE_STATUS:
                    wait = 2 ** attempt
                    log.warning(
                        "HTTP %d (attempt %d/%d) — waiting %ds",
                        response.status_code, attempt + 1, self._cfg.max_retries, wait,
                    )
                    time.sleep(wait)
                    last_exc = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    continue
                response.raise_for_status()  # raise on any remaining 4xx
                return response.json()
            except httpx.ConnectError as exc:
                wait = 2 ** attempt
                log.warning(
                    "Connection error (attempt %d/%d) — waiting %ds: %s",
                    attempt + 1, self._cfg.max_retries, wait, exc,
                )
                time.sleep(wait)
                last_exc = exc
            except httpx.TimeoutException as exc:
                wait = 2 ** attempt
                log.warning(
                    "Timeout (attempt %d/%d) — waiting %ds",
                    attempt + 1, self._cfg.max_retries, wait,
                )
                time.sleep(wait)
                last_exc = exc
            except httpx.HTTPStatusError:
                raise  # 4xx are not retried
        raise last_exc or RuntimeError("LLM call failed after all retries")

    def _parse_response(self, raw: dict[str, Any]) -> LLMResponse:
        choice = raw["choices"][0]
        msg = choice["message"]

        # Tool calls
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments")}
            tool_calls.append(ToolCall(
                call_id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=args,
            ))

        # Usage
        usage_data = raw.get("usage")
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
        else:
            # Estimate when the endpoint doesn't return usage
            estimated = len(str(msg.get("content") or "")) // _CHARS_PER_TOKEN
            usage = Usage(0, estimated, estimated)

        # Capture reasoning_content (Gemma 4 / vLLM thinking field).
        # This is stored in LLMResponse.thinking for the audit log ONLY.
        # It is never injected back into any agent message list.
        thinking: str | None = msg.get("reasoning_content") or None

        return LLMResponse(
            content=msg.get("content"),
            tool_calls=tool_calls,
            usage=usage,
            raw=raw,
            thinking=thinking,
        )

    def _audit_llm(self, resp: LLMResponse) -> None:
        """Write one LLM call to the audit log (no-op if no logger attached)."""
        if not self._audit:
            return
        try:
            ctx = self._call_ctx
            self._audit.log_llm(
                actor=ctx["actor"],
                tick_id=ctx["tick_id"],
                session_id=ctx["session_id"],
                goal_id=ctx["goal_id"],
                thinking=resp.thinking,
                content=resp.content,
                tool_calls=[
                    {"name": tc.name, "args": tc.arguments, "call_id": tc.call_id}
                    for tc in resp.tool_calls
                ],
                usage={
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                },
            )
        except Exception as exc:
            log.warning("_audit_llm failed: %s", exc)
