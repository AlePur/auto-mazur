"""
Configuration loading and validation.

All tunables live in config.yaml. Environment variables prefixed with
MAZUR_ override any value: MAZUR_MODEL, MAZUR_API_BASE, MAZUR_API_KEY, etc.
The api_key itself is read from the env var named by `api_key_env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    # ── LLM ───────────────────────────────────────────────────────────────
    model: str = "gpt-4o"
    api_base: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    max_retries: int = 3

    # ── Storage ───────────────────────────────────────────────────────────
    workspace_root: str = "./workspace"
    db_path: str = "./agent.db"

    # ── Tool limits ───────────────────────────────────────────────────────
    command_timeout_seconds: int = 300
    max_output_bytes: int = 102_400   # 100 KB
    max_read_bytes: int = 102_400

    # ── Worker session ────────────────────────────────────────────────────
    max_actions_per_session: int = 200
    max_consecutive_errors: int = 5
    context_compress_threshold_tokens: int = 60_000

    # ── Task loop ─────────────────────────────────────────────────────────
    max_task_attempts: int = 3

    # ── Consolidation schedule (ticks) ────────────────────────────────────
    checkpoint_interval: int = 50
    journal_interval: int = 500
    reflection_interval: int = 2_000
    weekly_summary_interval: int = 5_000
    archive_interval: int = 50_000

    # ── Health ────────────────────────────────────────────────────────────
    stuck_detection_window: int = 5
    failure_streak_threshold: int = 10
    neglect_threshold_ticks: int = 5_000

    # ── Derived (set after loading) ───────────────────────────────────────
    api_key: str = field(default="", repr=False)

    # ------------------------------------------------------------------
    def resolve_api_key(self) -> None:
        """Read the actual API key from the environment."""
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise EnvironmentError(
                f"API key env var '{self.api_key_env}' is not set. "
                "Export it before running the agent."
            )
        self.api_key = key

    def workspace_path(self) -> Path:
        return Path(self.workspace_root).resolve()

    def db_file(self) -> Path:
        return Path(self.db_path).resolve()


def load_config(path: str | Path = "config.yaml") -> Config:
    """
    Load config from YAML file, then apply MAZUR_* env var overrides.
    Finally resolves the API key from its named env var.
    """
    cfg = Config()

    yaml_path = Path(path)
    if yaml_path.exists():
        with yaml_path.open() as f:
            data: dict = yaml.safe_load(f) or {}
        _apply_dict(cfg, data)

    _apply_env_overrides(cfg)
    cfg.resolve_api_key()
    return cfg


# ── Helpers ────────────────────────────────────────────────────────────────

def _apply_dict(cfg: Config, data: dict) -> None:
    """Apply a dict of values to the Config dataclass, coercing types."""
    valid = {f.name: f for f in fields(cfg)}
    for key, value in data.items():
        if key in valid:
            f = valid[key]
            try:
                setattr(cfg, key, f.type(value) if isinstance(f.type, type) else value)
            except (TypeError, ValueError):
                setattr(cfg, key, value)


def _apply_env_overrides(cfg: Config) -> None:
    """
    Override config fields with MAZUR_<FIELD_NAME> environment variables.
    E.g. MAZUR_MODEL, MAZUR_API_BASE, MAZUR_MAX_RETRIES, …
    """
    valid = {f.name for f in fields(cfg)}
    for key, value in os.environ.items():
        if not key.startswith("MAZUR_"):
            continue
        field_name = key[len("MAZUR_"):].lower()
        if field_name in valid:
            # Coerce int fields
            current = getattr(cfg, field_name)
            if isinstance(current, int):
                try:
                    value = int(value)
                except ValueError:
                    pass
            setattr(cfg, field_name, value)
