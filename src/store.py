"""
Store — internal state management for the agent daemon.

All file operations that are part of the agent's *memory* (checkpoints,
journals, transcripts, meta documents) go through here.
These files are owned by the ``mazur`` daemon user and are NOT accessible
to the ``mazur-worker`` user that executes agent tool calls.  This
separation prevents the agent from reading, modifying, or deleting its own
internal state through its shell/read/write tools.

Knowledge is stored entirely in the database (knowledge_index + FTS5); there
are no knowledge files on disk.

Directory layout (relative to store_root, default /var/lib/mazur):
  goals/
    <goal_id>-<slug>/
      CHECKPOINT.md
      journal/
        <tick_start>-<tick_end>.md
      sessions/
        session-<id>.jsonl[.gz]
  meta/
    summaries/
      weekly-<tick>.md
  archive/
    ticks-before-<tick>.jsonl
"""

from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

# Top-level dirs created on init
_TOP_DIRS = [
    "goals",
    "meta",
    "meta/summaries",
    "archive",
]


class Store:
    """Manages all internal agent state on the filesystem.

    The store root is a separate directory from the agent's workspace
    (which the agent can see and modify).  Only the daemon process
    writes to the store; the agent's tools never touch it.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def ensure_structure(self) -> None:
        """Create the directory skeleton if it doesn't already exist."""
        for d in _TOP_DIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        log.debug("Store ready at %s", self.root)

    # ── Goal directories ───────────────────────────────────────────────────

    def goal_dir(self, goal_id: str) -> Path:
        """
        Return the Path for a goal's state directory.  Does NOT require it
        to exist.  The path key is Goal.workspace_path (a shared relative
        path used by both Store and Workspace).
        """
        return self.root / "goals" / goal_id

    def create_goal_state_dir(self, workspace_path: str) -> None:
        """
        Create the internal-state subdirectories for a new goal.
        ``workspace_path`` is the relative path returned by
        Workspace.create_goal_dir (e.g. ``goals/goal-001-slug``).
        """
        base = self.root / workspace_path
        for sub in ["", "journal", "sessions"]:
            (base / sub).mkdir(parents=True, exist_ok=True)
        log.info("Created goal state dir: %s", workspace_path)

    # ── Checkpoint ─────────────────────────────────────────────────────────

    def write_checkpoint(self, workspace_path: str, content: str) -> None:
        path = self.root / workspace_path / "CHECKPOINT.md"
        path.write_text(content, encoding="utf-8")

    def read_checkpoint(self, workspace_path: str) -> str | None:
        path = self.root / workspace_path / "CHECKPOINT.md"
        return path.read_text(encoding="utf-8") if path.exists() else None

    # ── Journal ────────────────────────────────────────────────────────────

    def append_journal(
        self, workspace_path: str, tick_start: int, tick_end: int, content: str
    ) -> Path:
        journal_dir = self.root / workspace_path / "journal"
        journal_dir.mkdir(exist_ok=True)
        path = journal_dir / f"{tick_start}-{tick_end}.md"
        path.write_text(content, encoding="utf-8")
        return path

    def read_journal_file(self, file_path: str) -> str | None:
        """Read a journal file by its store-relative path."""
        path = self.root / file_path
        return path.read_text(encoding="utf-8") if path.exists() else None

    def read_recent_journals(self, workspace_path: str, n: int) -> list[str]:
        """Return content of the n most-recent journal entries (oldest first).
        Legacy helper — prefer DB-backed get_recent_journals() + read_journal_file().
        """
        journal_dir = self.root / workspace_path / "journal"
        if not journal_dir.exists():
            return []
        files = sorted(journal_dir.glob("*.md"))[-n:]
        return [f.read_text(encoding="utf-8") for f in files]

    def list_journal_files(self, workspace_path: str) -> list[Path]:
        journal_dir = self.root / workspace_path / "journal"
        if not journal_dir.exists():
            return []
        return sorted(journal_dir.glob("*.md"))

    # ── Meta — Weekly summaries ────────────────────────────────────────────

    def write_weekly_summary(self, tick: int, content: str) -> Path:
        path = self.root / "meta" / "summaries" / f"weekly-{tick}.md"
        path.write_text(content, encoding="utf-8")
        return path

    def read_weekly_file(self, file_path: str) -> str | None:
        """Read a weekly summary file by its store-relative path."""
        path = self.root / file_path
        return path.read_text(encoding="utf-8") if path.exists() else None

    def read_weekly_summaries(self, n: int) -> list[str]:
        """Return content of the n most-recent weekly summaries (oldest first).
        Legacy helper — prefer DB-backed get_recent_weeklies() + read_weekly_file().
        """
        files = sorted((self.root / "meta" / "summaries").glob("weekly-*.md"))[-n:]
        return [f.read_text(encoding="utf-8") for f in files]

    # ── Session transcripts ────────────────────────────────────────────────

    def transcript_path(self, workspace_path: str, session_id: int) -> Path:
        return self.root / workspace_path / "sessions" / f"session-{session_id}.jsonl"

    def compress_transcript(self, path: Path) -> Path:
        """Gzip the transcript in place; removes original. Returns .gz path."""
        gz_path = path.with_suffix(".jsonl.gz")
        with path.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        path.unlink()
        log.info("Compressed transcript: %s", gz_path)
        return gz_path

    def delete_transcript(self, path: Path) -> None:
        """Delete a transcript (raw or compressed). Safe if file is missing."""
        for p in [path, path.with_suffix(".jsonl.gz")]:
            if p.exists():
                p.unlink()
                log.info("Deleted transcript: %s", p)

    # ── Generic helpers ────────────────────────────────────────────────────

    def read_file(self, rel_path: str, max_bytes: int = 0) -> str:
        """Read any file under store root, with optional byte cap."""
        path = self.root / rel_path
        text = path.read_text(encoding="utf-8", errors="replace")
        if max_bytes and len(text.encode()) > max_bytes:
            text = text.encode()[:max_bytes].decode(errors="replace")
            text += f"\n[truncated — file exceeds {max_bytes} bytes]"
        return text

    def write_file(self, rel_path: str, content: str) -> None:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def abs(self, rel_path: str) -> Path:
        """Resolve a store-relative path to an absolute Path."""
        return self.root / rel_path
