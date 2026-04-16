"""
Tool implementations — the Worker's interface to the real world.

Four tools:
  shell(command)              → run any shell command via bash
  read(path, lines="0-100")  → read a file, optionally a line range
  write(path, content)        → write a file
  finish(summary, status)     → end the session

All three I/O tools (shell, read, write) run as the ``worker_user`` when
configured.  This means they share the same filesystem permissions — only
paths accessible to the worker user are reachable.  The daemon's internal
state (Store) is owned by a different user and is invisible to these tools.

When ``worker_user`` is empty (local development), all tools run as the
current process user — no sudo involved.

The read tool applies a line-range default (config.max_read_lines, default 500)
and an optional hard character cap (config.max_read_chars; 0 = disabled).

Output always includes file metadata so the Worker knows the full file size
and can issue follow-up reads with a different line range.

Shell output is truncated at config.max_output_bytes.

finish() is handled by the execution loop (session.py), not here.
"""

from __future__ import annotations

import json
import logging
import select
import subprocess
import time
import uuid
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
                "By default returns the first 500 lines. "
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
                            "Omit to get the default first 500 lines."
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


class PersistentShell:
    """
    A single bash login shell kept alive for the duration of a WorkerSession.

    Commands are written to its stdin and output is read until a unique
    sentinel marker, so the shell process is never torn down between calls.
    This means cwd, environment variables, shell functions, activated Python
    venvs, etc. all persist across every shell() call within one session —
    exactly the behaviour an LLM agent expects.

    The --login flag sources /etc/profile and the per-user profile, giving
    the worker user access to all packages listed in its user-level packages.

    When ``worker_user`` is set, the shell is spawned via
    ``sudo -u <worker_user> bash --login`` so that all commands run with
    the worker user's permissions.  When empty, plain ``bash --login`` is
    used (local development).
    """

    def __init__(self, worker_user: str = "", initial_cwd: str = "") -> None:
        # A unique, unguessable sentinel that cannot plausibly appear in
        # normal command output.
        self._sentinel = f"__MAZUR_{uuid.uuid4().hex}__"
        self._worker_user = worker_user
        self._initial_cwd = initial_cwd
        self._proc: subprocess.Popen | None = None
        self._start()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def _start(self) -> None:
        """Spawn (or re-spawn) the persistent bash login shell."""
        if self._worker_user:
            cmd = ["sudo", "-u", self._worker_user, "bash", "--login"]
        else:
            cmd = ["bash", "--login"]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
        )
        # Suppress the prompt and command echo so our sentinel parsing is clean.
        # Then cd to the goal's work directory so the agent starts there.
        init = "PS1=''; PS2=''; set +o history\n"
        if self._initial_cwd:
            import shlex
            init += f"cd {shlex.quote(self._initial_cwd)}\n"
        assert self._proc.stdin is not None
        self._proc.stdin.write(init)
        self._proc.stdin.flush()
        log.debug(
            "PersistentShell: started pid=%d%s%s",
            self._proc.pid,
            f" (as {self._worker_user})" if self._worker_user else "",
            f" (cwd={self._initial_cwd})" if self._initial_cwd else "",
        )

    def close(self) -> None:
        """Gracefully shut down the shell process."""
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                self._proc.stdin.write("exit\n")  # type: ignore[union-attr]
                self._proc.stdin.flush()           # type: ignore[union-attr]
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=3)
                except Exception:
                    self._proc.kill()
        self._proc = None
        log.debug("PersistentShell: closed")

    # ── Command execution ──────────────────────────────────────────────────

    def run(self, command: str, timeout: int) -> tuple[str, int, bool]:
        """
        Run *command* inside the persistent shell.

        Returns ``(output, returncode, timed_out)``.

        If the underlying bash process has died between calls it is
        automatically restarted (state is lost, but we don't crash).
        """
        # Auto-restart if the process died
        if self._proc is None or self._proc.poll() is not None:
            log.warning("PersistentShell: bash process died, restarting")
            self._start()

        sentinel = self._sentinel
        # Write the command followed by a line that prints the sentinel and
        # the exit code.  printf avoids any flag interpretation issues.
        wrapped = f"{command}\n__mc=$?; printf '%s %d\\n' '{sentinel}' $__mc\n"

        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        try:
            self._proc.stdin.write(wrapped)
            self._proc.stdin.flush()
        except BrokenPipeError:
            log.warning("PersistentShell: stdin broken, restarting")
            self._start()
            assert self._proc is not None
            assert self._proc.stdin is not None
            self._proc.stdin.write(wrapped)
            self._proc.stdin.flush()

        # Read output line-by-line until we see the sentinel, subject to the
        # per-command timeout.  select() lets us do a non-blocking wait so we
        # can honour the deadline precisely.
        lines: list[str] = []
        deadline = time.monotonic() + timeout
        timed_out = False
        rc = 0

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break

            ready, _, _ = select.select([self._proc.stdout], [], [], remaining)
            if not ready:
                timed_out = True
                # Kill and restart the shell so stale output from this timed-out
                # command cannot bleed into the next run() call.  The sentinel
                # for *this* command is still sitting in the pipe buffer; if we
                # left the process alive, the next run() would read those stale
                # lines and mistake them for the new command's output.
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=3)
                except Exception:
                    pass
                self._proc = None
                self._start()
                break

            line = self._proc.stdout.readline()
            if not line:
                # EOF — process exited unexpectedly
                self._proc = None
                break

            stripped = line.rstrip("\n")
            # Sentinel line looks like:  __MAZUR_<hex>__ <returncode>
            if stripped.startswith(sentinel):
                try:
                    rc = int(stripped.split()[-1])
                except (ValueError, IndexError):
                    rc = 0
                break

            lines.append(stripped)

        return "\n".join(lines), rc, timed_out


class ToolExecutor:
    """
    Executes Worker tool calls.  One instance is shared across the lifetime
    of a WorkerSession.

    A single PersistentShell is kept alive for the whole session so that
    shell state (cwd, env-vars, venvs, etc.) carries across tool calls.
    Call close() when the session ends to tear it down.

    When ``config.worker_user`` is set, all three I/O tools (shell, read,
    write) run as that user — providing uniform filesystem permissions
    that match the agent's workspace and exclude the daemon's internal
    state.
    """

    def __init__(self, config: Config, initial_cwd: str = "") -> None:
        self._cfg = config
        self._initial_cwd = initial_cwd or None
        self._shell = PersistentShell(worker_user=config.worker_user, initial_cwd=initial_cwd)

    def close(self) -> None:
        """Shut down the persistent bash shell.  Call once per session."""
        self._shell.close()

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

    # ── Helper: run a command as the worker user ───────────────────────────

    def _run_as_worker(
        self,
        args: list[str],
        input: bytes | None = None,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess:
        """
        Run a command, optionally as the configured worker user via sudo.
        When worker_user is empty, runs directly as the current user.
        """
        if self._cfg.worker_user:
            args = ["sudo", "-n", "-u", self._cfg.worker_user, "--"] + args
        return subprocess.run(
            args,
            input=input,
            capture_output=True,
            timeout=timeout,
            cwd=self._initial_cwd,
        )

    # ── Individual tools ───────────────────────────────────────────────────

    def shell(self, command: str) -> ToolResult:
        if not command.strip():
            return ToolResult(output="[empty command]", is_error=True, truncated=False)

        _parts = command.strip().split()
        if _parts[0] == "ls" and any("R" in p for p in _parts[1:] if p.startswith("-")):
            return ToolResult(
                output=(
                    "ls -R will lead to context flooding or shell timeout "
                    "in a directory with many files"
                ),
                is_error=True,
                truncated=False,
            )

        log.debug("shell: %s", command[:200])
        try:
            output, rc, timed_out = self._shell.run(
                command, self._cfg.command_timeout_seconds
            )
            if timed_out:
                return ToolResult(
                    output=(
                        f"[TIMEOUT — command killed after "
                        f"{self._cfg.command_timeout_seconds}s]\n"
                        "Use non-blocking commands or run long jobs in the background."
                    ),
                    is_error=True,
                    truncated=False,
                )
            is_error = rc != 0
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

        Runs as the worker user when configured, so only files accessible
        to that user can be read.
        """
        if not path.strip():
            return ToolResult(output="[empty path]", is_error=True, truncated=False)

        log.debug("read: %s lines=%s", path, lines)

        try:
            proc = self._run_as_worker(["cat", "--", path], timeout=30)
            if proc.returncode != 0:
                stderr = proc.stderr.decode(errors="replace").strip()
                if "No such file" in stderr:
                    return ToolResult(
                        output=f"[FILE NOT FOUND: {path}]",
                        is_error=True,
                        truncated=False,
                    )
                elif "Permission denied" in stderr:
                    return ToolResult(
                        output=f"[PERMISSION DENIED: {path}]",
                        is_error=True,
                        truncated=False,
                    )
                else:
                    return ToolResult(
                        output=f"[ERROR reading {path}: {stderr}]",
                        is_error=True,
                        truncated=False,
                    )
            raw_bytes = proc.stdout
        except subprocess.TimeoutExpired:
            return ToolResult(
                output=f"[TIMEOUT reading {path}]",
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
        """
        Write content to a file.

        Runs as the worker user when configured, so only paths writable
        by that user can be written to.
        """
        if not path.strip():
            return ToolResult(output="[empty path]", is_error=True, truncated=False)

        log.debug("write: %s (%d chars)", path, len(content))

        try:
            # Create parent directories
            parent = str(Path(path).parent)
            self._run_as_worker(["mkdir", "-p", "--", parent], timeout=10)

            # Write content via tee (stdin → file)
            proc = self._run_as_worker(
                ["tee", "--", path],
                input=content.encode("utf-8"),
                timeout=30,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode(errors="replace").strip()
                if "Permission denied" in stderr:
                    return ToolResult(
                        output=f"[PERMISSION DENIED: {path}]",
                        is_error=True,
                        truncated=False,
                    )
                return ToolResult(
                    output=f"[ERROR writing {path}: {stderr}]",
                    is_error=True,
                    truncated=False,
                )
        except subprocess.TimeoutExpired:
            return ToolResult(
                output=f"[TIMEOUT writing {path}]",
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
