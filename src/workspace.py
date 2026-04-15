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
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def ensure_structure(self) -> None:
        """Create the directory skeleton if it doesn't already exist."""
        for d in _TOP_DIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)
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
            (base / sub).mkdir(parents=True, exist_ok=True)
        log.info("Created goal work dir: %s", rel)
        return rel

    def goal_work_dir(self, workspace_path: str) -> Path:
        """Return the absolute path to a goal's work directory."""
        return self.root / workspace_path

    # ── Generic helpers ────────────────────────────────────────────────────

    def abs(self, rel_path: str) -> Path:
        """Resolve a workspace-relative path to an absolute Path."""
        return self.root / rel_path


# ── Internal helpers ───────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")
