"""
LLM client — thin wrapper around the OpenAI-compatible chat completions API.

Responsibilities:
  - Retry with exponential backoff on transient errors
  - Normalise responses into our internal LLMResponse type
  - Rough token counting for context-budget tracking
  - No prompt logic lives here; that belongs to characters/

Supports any OpenAI-compatible endpoint: OpenAI, Azure, Ollama, vLLM, etc.
Set config.api_base and config.model accordingly.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import openai

from .config import Config
from .models import LLMResponse, ToolCall, Usage

log = logging.getLogger(__name__)

# Rough token estimate: 1 token ≈ 4 chars for most LLMs.
# Used for budget tracking when the API doesn't return usage.
_CHARS_PER_TOKEN = 4


class LLMClient:
    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
        )

    # ── Main interface ─────────────────────────────────────────────────────

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
        - tool_choice: "auto" | "none" | {"type": "function", "function": {"name": "..."}}
          Defaults to "auto" when tools are provided, otherwise omitted.
        """
        kwargs: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        raw = self._call_with_retry(**kwargs)
        return self._parse_response(raw)

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """
        Like chat() but instructs the model to return valid JSON and
        parses the response.  Used for structured output (Reflector, etc.)
        """
        kwargs: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        raw = self._call_with_retry(**kwargs)
        resp = self._parse_response(raw)
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

    def _call_with_retry(self, **kwargs) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self._cfg.max_retries):
            try:
                return self._client.chat.completions.create(**kwargs)
            except openai.RateLimitError as exc:
                wait = 2 ** attempt
                log.warning("Rate limit hit (attempt %d/%d) — waiting %ds",
                            attempt + 1, self._cfg.max_retries, wait)
                time.sleep(wait)
                last_exc = exc
            except openai.APIStatusError as exc:
                if exc.status_code >= 500:
                    wait = 2 ** attempt
                    log.warning("Server error %d (attempt %d/%d) — waiting %ds",
                                exc.status_code, attempt + 1, self._cfg.max_retries, wait)
                    time.sleep(wait)
                    last_exc = exc
                else:
                    raise  # 4xx are not retried
            except openai.APIConnectionError as exc:
                wait = 2 ** attempt
                log.warning("Connection error (attempt %d/%d) — waiting %ds",
                            attempt + 1, self._cfg.max_retries, wait)
                time.sleep(wait)
                last_exc = exc
        raise last_exc or RuntimeError("LLM call failed after all retries")

    def _parse_response(self, raw: Any) -> LLMResponse:
        choice = raw.choices[0]
        msg = choice.message

        # Tool calls
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    call_id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        # Usage
        if raw.usage:
            usage = Usage(
                prompt_tokens=raw.usage.prompt_tokens,
                completion_tokens=raw.usage.completion_tokens,
                total_tokens=raw.usage.total_tokens,
            )
        else:
            # Estimate when the endpoint doesn't return usage
            estimated = len(str(msg.content or "")) // _CHARS_PER_TOKEN
            usage = Usage(0, estimated, estimated)

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            usage=usage,
            raw=raw,
        )
