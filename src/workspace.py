"""
Workspace — the agent's visible working area.

This is the directory tree that the agent can see and modify through its
shell/read/write tools.  It contains ONLY work products — source code,
data files, and scratch space.  No internal state (checkpoints, journals,
knowledge, audit, transcripts) lives here; that is managed by the Store
class which the agent cannot access.

Directory layout (relative to workspace_root, default /home/mazur-worker):
  goals/
    <goal_id>-<slug>/
      src/
      data/
  scratch/
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

# Top-level dirs created on init
_TOP_DIRS = [
    "goals",
    "scratch",
]


class Workspace:
    """Manages the agent-facing work directories.

    Everything under the workspace root is accessible to the agent
    through its shell, read, and write tools.  The workspace contains
    only things the agent is meant to work with — goal work directories
    (src/, data/) and a scratch area.

    When *worker_user* is set (production mode), directories are created
    via ``sudo -u <worker_user> mkdir -p`` so they are owned by the
    worker user and the agent's tools can write to them.  In local
    development (no worker_user) directories are created directly.
    """

    def __init__(self, root: str | Path, worker_user: str = "") -> None:
        self.root = Path(root).resolve()
        self._worker_user = worker_user

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def ensure_structure(self) -> None:
        """Create the directory skeleton if it doesn't already exist."""
        for d in _TOP_DIRS:
            self._mkdir(self.root / d)
        log.debug("Workspace ready at %s", self.root)

    # ── Goal work directories ──────────────────────────────────────────────

    def create_goal_dir(self, goal_id: str, title_slug: str) -> str:
        """
        Create the work-area directory structure for a new goal.
        Returns the relative path (used as Goal.workspace_path — shared
        with the Store to locate the corresponding state directory).
        """
        safe_slug = _slugify(title_slug)[:40]
        rel = f"goals/{goal_id}-{safe_slug}"
        base = self.root / rel
        for sub in ["", "src", "data"]:
            self._mkdir(base / sub)
        log.info("Created goal work dir: %s", rel)
        return rel

    def goal_work_dir(self, workspace_path: str) -> Path:
        """Return the absolute path to a goal's work directory."""
        return self.root / workspace_path

    # ── Generic helpers ────────────────────────────────────────────────────

    def abs(self, rel_path: str) -> Path:
        """Resolve a workspace-relative path to an absolute Path."""
        return self.root / rel_path


    # ── Private helpers ───────────────────────────────────────────────────

    def _mkdir(self, path: Path) -> None:
        """Create *path* (and any missing parents) with the right ownership.

        In production (*worker_user* set) the directory is created via
        ``sudo -u <worker_user> mkdir -p`` so it is owned by the worker
        user.  In local dev the directory is created directly by the
        current process.
        """
        if self._worker_user:
            subprocess.run(
                ["sudo", "-u", self._worker_user, "mkdir", "-p", str(path)],
                check=True,
            )
        else:
            path.mkdir(parents=True, exist_ok=True)


# ── Internal helpers ───────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")
