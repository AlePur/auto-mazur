"""
Configuration loading and validation.

All tunables live in config.yaml. Environment variables prefixed with
MAZUR_ override any value: MAZUR_MODEL, MAZUR_API_BASE, MAZUR_MAX_RETRIES, etc.

No API key is required: the vLLM server on Tailscale is accessed without
authentication (local-network only, no public exposure).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

# Minimum context window we are willing to run with.
_MIN_CONTEXT_LENGTH_TOKENS = 100_000


@dataclass
class Config:
    # ── LLM ───────────────────────────────────────────────────────────────
    model: str = "google/gemma-4-31B-it"
    api_base: str = "http://localhost:8000/v1"
    max_retries: int = 3

    # ── Context window ────────────────────────────────────────────────────
    # Must be >= 100 000 — validated at load time.
    context_length_tokens: int = 128_000

    # ── Storage ───────────────────────────────────────────────────────────
    workspace_root: str = "./workspace"
    store_root: str = "./store"
    db_path: str = "./agent.db"

    # ── User separation ───────────────────────────────────────────────────
    # When set, shell/read/write tools run as this user (via sudo).
    # Empty string = run as the current user (local development mode).
    worker_user: str = ""

    # ── Tool limits ───────────────────────────────────────────────────────
    command_timeout_seconds: int = 300
    max_output_bytes: int = 102_400   # 100 KB for shell output
    max_read_bytes: int = 102_400     # 100 KB raw byte cap for file reads
    max_read_chars: int = 30_000      # hard char cap for read tool (~7 500 tokens); 0 = use bytes cap only
    max_read_lines: int = 100         # default lines returned when no range given

    # ── Worker session ────────────────────────────────────────────────────
    max_actions_per_session: int = 200
    max_consecutive_errors: int = 5
    # 0 = auto (60 % of context_length_tokens). Set explicitly to override.
    context_compress_threshold_tokens: int = 0

    # ── Executive query loop ──────────────────────────────────────────────
    max_executive_queries: int = 10

    # ── Task loop ─────────────────────────────────────────────────────────
    max_task_attempts: int = 3

    # ── Consolidation schedule (ticks) ────────────────────────────────────
    checkpoint_interval: int = 50
    # Auto-journal a goal when it accumulates this many ticks since last journal.
    journal_activity_threshold: int = 100
    weekly_summary_interval: int = 5_000
    archive_interval: int = 50_000

    # ── Health ────────────────────────────────────────────────────────────
    stuck_detection_window: int = 5
    failure_streak_threshold: int = 10
    neglect_threshold_ticks: int = 5_000

    # ── Gateway HTTP server ────────────────────────────────────────────────
    # Set gateway_enabled: true in config.yaml (or MAZUR_GATEWAY_ENABLED=true)
    # to start the observation/inbox HTTP server alongside the agent loop.
    gateway_enabled: bool = False
    gateway_host: str = "127.0.0.1"
    gateway_port: int = 7878

    def effective_compress_threshold(self) -> int:
        """
        Return the effective compression threshold.

        If context_compress_threshold_tokens is 0 (auto), derive it as
        60 % of context_length_tokens.  Otherwise use the explicit value,
        capped at context_length_tokens so it can never exceed the window.
        """
        if self.context_compress_threshold_tokens == 0:
            return int(self.context_length_tokens * 0.6)
        return min(self.context_compress_threshold_tokens, self.context_length_tokens)

    def workspace_path(self) -> Path:
        return Path(self.workspace_root).resolve()

    def store_path(self) -> Path:
        return Path(self.store_root).resolve()

    def db_file(self) -> Path:
        return Path(self.db_path).resolve()


def load_config(path: str | Path = "config.yaml") -> Config:
    """
    Load config from YAML file, then apply MAZUR_* env var overrides
    and validate critical constraints.
    """
    cfg = Config()

    yaml_path = Path(path)
    if yaml_path.exists():
        with yaml_path.open() as f:
            data: dict = yaml.safe_load(f) or {}
        _apply_dict(cfg, data)

    _apply_env_overrides(cfg)
    _validate(cfg)
    return cfg


# ── Helpers ────────────────────────────────────────────────────────────────

def _validate(cfg: Config) -> None:
    """Raise on configuration that would cause the agent to malfunction."""
    if cfg.context_length_tokens < _MIN_CONTEXT_LENGTH_TOKENS:
        raise ValueError(
            f"context_length_tokens is {cfg.context_length_tokens:,} but the minimum "
            f"is {_MIN_CONTEXT_LENGTH_TOKENS:,}.  Update config.yaml or set "
            f"MAZUR_CONTEXT_LENGTH_TOKENS to at least {_MIN_CONTEXT_LENGTH_TOKENS:,}."
        )


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
            current = getattr(cfg, field_name)
            # Coerce int fields
            if isinstance(current, int):
                try:
                    value = int(value)
                except ValueError:
                    pass
            # Coerce bool fields (true/1/yes → True, anything else → False)
            elif isinstance(current, bool):
                value = value.strip().lower() in ("1", "true", "yes", "on")
            setattr(cfg, field_name, value)
