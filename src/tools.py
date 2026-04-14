"""
Tool implementations — the Worker's interface to the real world.

Three tools, nothing more:
  shell(command)         → run any shell command via bash
  read(path)             → read a file
  write(path, content)   → write a file

The only constraints are engineering ones that protect the infrastructure:
  - Commands are killed after config.command_timeout_seconds
  - Output is truncated at config.max_output_bytes (can't fit 10MB in context)
  - File reads are truncated at config.max_read_bytes

No deny-lists, no path jails, no content filtering.
The OS user's permissions are the only security boundary.

finish() is handled by the execution loop (session.py), not here.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from .config import Config
from .models import ToolResult

log = logging.getLogger(__name__)


# ── OpenAI function-calling schemas for the Worker's tools ────────────────
# These are passed verbatim to the LLM in the tools parameter.

WORKER_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": (
                "Run any bash shell command. "
                "Returns combined stdout+stderr. "
                "Commands that produce no output return '(no output)'. "
                "Long output is truncated with a note at the end."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to run.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read the contents of a file at the given path. "
                "Very large files are truncated with a note."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": (
                "Write content to a file at the given path. "
                "Creates parent directories as needed. "
                "Overwrites any existing file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to write to.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Signal that the task is complete or that you are stuck. "
                "Call this when the task criteria are met, or when you cannot "
                "make further progress without external help."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "1-3 sentences: what was accomplished, "
                            "what state things are in, any blockers."
                        ),
                    },
                    "status": {
                        "type": "string",
                        "enum": ["done", "stuck", "error"],
                        "description": (
                            "'done' — task criteria met. "
                            "'stuck' — blocked, need replanning. "
                            "'error' — unrecoverable error state."
                        ),
                    },
                },
                "required": ["summary", "status"],
            },
        },
    },
]


class ToolExecutor:
    """
    Executes Worker tool calls.  One instance is shared across the lifetime
    of a WorkerSession.
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Dispatch a tool call by name and return a ToolResult."""
        match name:
            case "shell":
                return self.shell(arguments.get("command", ""))
            case "read":
                return self.read(arguments.get("path", ""))
            case "write":
                return self.write(
                    arguments.get("path", ""),
                    arguments.get("content", ""),
                )
            case "finish":
                # finish() is intercepted by the session loop before reaching
                # here, but we handle it gracefully just in case.
                return ToolResult(
                    output="[finish signalled]",
                    is_error=False,
                    truncated=False,
                )
            case _:
                return ToolResult(
                    output=f"[unknown tool: {name!r}]",
                    is_error=True,
                    truncated=False,
                )

    # ── Individual tools ───────────────────────────────────────────────────

    def shell(self, command: str) -> ToolResult:
        if not command.strip():
            return ToolResult(output="[empty command]", is_error=True, truncated=False)

        log.debug("shell: %s", command[:200])
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=self._cfg.command_timeout_seconds,
            )
            output = proc.stdout + proc.stderr
            is_error = proc.returncode != 0
        except subprocess.TimeoutExpired:
            return ToolResult(
                output=(
                    f"[TIMEOUT — command killed after "
                    f"{self._cfg.command_timeout_seconds}s]\n"
                    "Use non-blocking commands or run long jobs in the background."
                ),
                is_error=True,
                truncated=False,
            )
        except Exception as exc:
            return ToolResult(
                output=f"[ERROR launching command: {exc}]",
                is_error=True,
                truncated=False,
            )

        truncated = False
        if len(output.encode()) > self._cfg.max_output_bytes:
            original_len = len(output.encode())
            output = output.encode()[: self._cfg.max_output_bytes].decode(errors="replace")
            output += (
                f"\n[TRUNCATED — full output was {original_len:,} bytes; "
                f"showing first {self._cfg.max_output_bytes:,} bytes]"
            )
            truncated = True

        if not output:
            output = "(no output)"

        return ToolResult(output=output, is_error=is_error, truncated=truncated)

    def read(self, path: str) -> ToolResult:
        if not path.strip():
            return ToolResult(output="[empty path]", is_error=True, truncated=False)

        p = Path(path)
        log.debug("read: %s", p)
        try:
            raw = p.read_bytes()
        except FileNotFoundError:
            return ToolResult(
                output=f"[FILE NOT FOUND: {path}]",
                is_error=True,
                truncated=False,
            )
        except PermissionError:
            return ToolResult(
                output=f"[PERMISSION DENIED: {path}]",
                is_error=True,
                truncated=False,
            )
        except Exception as exc:
            return ToolResult(
                output=f"[ERROR reading {path}: {exc}]",
                is_error=True,
                truncated=False,
            )

        truncated = False
        if len(raw) > self._cfg.max_read_bytes:
            original_len = len(raw)
            raw = raw[: self._cfg.max_read_bytes]
            text = raw.decode(errors="replace")
            text += (
                f"\n[TRUNCATED — file is {original_len:,} bytes; "
                f"showing first {self._cfg.max_read_bytes:,} bytes. "
                "Use shell('grep pattern file') or shell('head -n N file') "
                "for targeted access.]"
            )
            truncated = True
        else:
            text = raw.decode(errors="replace")

        return ToolResult(output=text, is_error=False, truncated=truncated)

    def write(self, path: str, content: str) -> ToolResult:
        if not path.strip():
            return ToolResult(output="[empty path]", is_error=True, truncated=False)

        p = Path(path)
        log.debug("write: %s (%d chars)", p, len(content))
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except PermissionError:
            return ToolResult(
                output=f"[PERMISSION DENIED: {path}]",
                is_error=True,
                truncated=False,
            )
        except Exception as exc:
            return ToolResult(
                output=f"[ERROR writing {path}: {exc}]",
                is_error=True,
                truncated=False,
            )

        return ToolResult(
            output=f"Written {len(content):,} chars to {path}",
            is_error=False,
            truncated=False,
        )


def format_tool_result_for_llm(
    call_id: str,
    result: ToolResult,
) -> dict[str, Any]:
    """
    Format a ToolResult as an OpenAI tool-result message dict.
    """
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": result.output,
    }


def format_tool_call_for_transcript(
    name: str,
    arguments: dict[str, Any],
    result: ToolResult,
    tick_id: int,
) -> str:
    """
    Serialise a single tool call + result to a JSONL line for the transcript.
    """
    return json.dumps({
        "tick": tick_id,
        "tool": name,
        "args": arguments,
        "output": result.output[:2000],  # cap transcript entries at 2KB each
        "truncated": result.truncated,
        "error": result.is_error,
    })
