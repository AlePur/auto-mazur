"""
Tool implementations — the Worker's interface to the real world.

Four tools:
  shell(command)              → run any shell command via bash
  read(path, lines="0-100")  → read a file, optionally a line range
  write(path, content)       → write a file
  finish(summary, status)    → end the session

The read tool always applies two caps (whichever is more restrictive):
  - lines range (default first 100 lines, configurable via config.max_read_lines)
  - hard character cap (config.max_read_chars, ~7 500 tokens)

Output always includes file metadata so the Worker knows the full file size
and can issue follow-up reads with a different line range.

Shell output is truncated at config.max_output_bytes.

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
                "Read the contents of a file. "
                "By default returns the first 100 lines. "
                "Use the `lines` parameter to read a specific range, e.g. '100-200'. "
                "Output always includes a header showing total file size so you can "
                "plan follow-up reads for large files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                    "lines": {
                        "type": "string",
                        "description": (
                            "Line range to return, zero-indexed, inclusive. "
                            "Format: 'START-END', e.g. '0-100', '200-300', '500-600'. "
                            "Omit to get the default first 100 lines."
                        ),
                    },
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
                return self.read(
                    arguments.get("path", ""),
                    arguments.get("lines", None),
                )
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

    def read(self, path: str, lines: str | None = None) -> ToolResult:
        """
        Read a file, optionally a specific line range.

        `lines` format: "START-END" (zero-indexed, inclusive).
        Default: first config.max_read_lines lines.

        Output is always capped at config.max_read_chars characters.
        A metadata header is prepended so the Worker knows the full file size.
        """
        if not path.strip():
            return ToolResult(output="[empty path]", is_error=True, truncated=False)

        p = Path(path)
        log.debug("read: %s lines=%s", p, lines)

        try:
            raw_bytes = p.read_bytes()
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

        # Decode full text for line-based slicing
        full_text = raw_bytes.decode(errors="replace")
        all_lines = full_text.splitlines(keepends=True)
        total_lines = len(all_lines)
        total_bytes = len(raw_bytes)

        # Parse line range
        default_end = self._cfg.max_read_lines - 1  # 0-indexed inclusive
        line_start, line_end = _parse_line_range(lines, default_end)

        # Clamp to actual file length
        line_start = max(0, min(line_start, total_lines))
        line_end = max(line_start, min(line_end, total_lines - 1))

        selected_lines = all_lines[line_start : line_end + 1]
        text = "".join(selected_lines)

        # Apply hard character cap
        char_cap = self._cfg.max_read_chars
        truncated = False
        if char_cap > 0 and len(text) > char_cap:
            text = text[:char_cap]
            truncated = True

        # Build metadata header
        showing_lines = f"{line_start}-{line_end}" if total_lines > 0 else "0-0"
        header = (
            f"[File: {path} | "
            f"Total: {total_lines} lines / {total_bytes:,} bytes | "
            f"Showing lines {showing_lines}"
        )
        if truncated:
            header += f" | TRUNCATED at {char_cap:,} chars"
        header += "]\n"

        return ToolResult(
            output=header + text,
            is_error=False,
            truncated=truncated,
        )

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


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_line_range(lines: str | None, default_end: int) -> tuple[int, int]:
    """
    Parse a 'START-END' line range string.
    Returns (start, end) as zero-indexed inclusive integers.
    Falls back to (0, default_end) on any parse error.
    """
    if not lines:
        return 0, default_end

    lines = lines.strip()
    try:
        if "-" in lines:
            parts = lines.split("-", 1)
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            if start < 0 or end < start:
                return 0, default_end
            return start, end
        else:
            # Single number — treat as end line
            end = int(lines)
            return 0, max(0, end)
    except (ValueError, IndexError):
        return 0, default_end
