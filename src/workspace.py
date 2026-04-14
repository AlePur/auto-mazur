"""
Workspace — filesystem management for the agent's home directory.

All file operations that are part of the agent's *memory* (checkpoints,
journals, knowledge, transcripts, meta documents) go through here.
The Worker's raw tool calls (shell/read/write for arbitrary paths) are
in tools.py — those are not mediated by the workspace layer.

Directory layout (relative to workspace_root):
  goals/
    <goal_id>-<slug>/
      PLAN.md
      STATUS.md
      CHECKPOINT.md
      journal/
        <tick_start>-<tick_end>.md
      sessions/
        session-<id>.jsonl[.gz]
      src/
      data/
  knowledge/
    <topic>.md
  meta/
    PRIORITIES.md
    REFLECTIONS.md
    summaries/
      weekly-<tick>.md
      monthly-<tick>.md
  scratch/
  archive/
    ticks-<start>-<end>.jsonl.gz
"""

from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

# Top-level dirs created on init
_TOP_DIRS = ["goals", "knowledge", "meta", "meta/summaries", "scratch", "archive"]


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def ensure_structure(self) -> None:
        """Create the directory skeleton if it doesn't already exist."""
        for d in _TOP_DIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        log.debug("Workspace ready at %s", self.root)

    # ── Goal directories ───────────────────────────────────────────────────

    def goal_dir(self, goal_id: str) -> Path:
        """
        Return the Path for a goal directory.  Does NOT require it to exist.
        The path is stored in Goal.workspace_path; use that as the key.
        """
        # goal workspace_path is stored relative to workspace root
        # If the caller has an absolute path use it directly, else resolve
        return self.root / "goals" / goal_id

    def create_goal_dir(self, goal_id: str, title_slug: str) -> str:
        """
        Create the directory structure for a new goal.
        Returns the relative path (used as Goal.workspace_path).
        """
        safe_slug = _slugify(title_slug)[:40]
        rel = f"goals/{goal_id}-{safe_slug}"
        base = self.root / rel
        for sub in ["", "journal", "sessions", "src", "data"]:
            (base / sub).mkdir(parents=True, exist_ok=True)
        log.info("Created goal dir: %s", rel)
        return rel

    # ── Checkpoint ─────────────────────────────────────────────────────────

    def write_checkpoint(self, workspace_path: str, content: str) -> None:
        path = self.root / workspace_path / "CHECKPOINT.md"
        path.write_text(content, encoding="utf-8")

    def read_checkpoint(self, workspace_path: str) -> str | None:
        path = self.root / workspace_path / "CHECKPOINT.md"
        return path.read_text(encoding="utf-8") if path.exists() else None

    # ── Plan / Status ──────────────────────────────────────────────────────

    def write_plan(self, workspace_path: str, content: str) -> None:
        (self.root / workspace_path / "PLAN.md").write_text(content, encoding="utf-8")

    def read_plan(self, workspace_path: str) -> str | None:
        path = self.root / workspace_path / "PLAN.md"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def write_status(self, workspace_path: str, content: str) -> None:
        (self.root / workspace_path / "STATUS.md").write_text(content, encoding="utf-8")

    def read_status(self, workspace_path: str) -> str | None:
        path = self.root / workspace_path / "STATUS.md"
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

    def read_recent_journals(self, workspace_path: str, n: int) -> list[str]:
        """Return content of the n most-recent journal entries (oldest first)."""
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

    # ── Knowledge ──────────────────────────────────────────────────────────

    def write_knowledge(self, topic: str, content: str) -> Path:
        path = self.root / "knowledge" / f"{topic}.md"
        path.write_text(content, encoding="utf-8")
        return path

    def read_knowledge(self, topic: str) -> str | None:
        path = self.root / "knowledge" / f"{topic}.md"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def list_knowledge_files(self) -> list[Path]:
        return sorted((self.root / "knowledge").glob("*.md"))

    # ── Meta ───────────────────────────────────────────────────────────────

    def write_priorities(self, content: str) -> None:
        (self.root / "meta" / "PRIORITIES.md").write_text(content, encoding="utf-8")

    def read_priorities(self) -> str | None:
        path = self.root / "meta" / "PRIORITIES.md"
        return path.read_text(encoding="utf-8") if path.exists() else None

    def append_reflection(self, content: str) -> None:
        path = self.root / "meta" / "REFLECTIONS.md"
        with path.open("a", encoding="utf-8") as f:
            f.write("\n\n---\n\n" + content)

    def write_weekly_summary(self, tick: int, content: str) -> Path:
        path = self.root / "meta" / "summaries" / f"weekly-{tick}.md"
        path.write_text(content, encoding="utf-8")
        return path

    def read_weekly_summaries(self, n: int) -> list[str]:
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
        """Read any file under workspace root, with optional byte cap."""
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
        """Resolve a workspace-relative path to an absolute Path."""
        return self.root / rel_path


# ── Internal helpers ───────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")
