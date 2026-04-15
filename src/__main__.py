"""
Entry point: python -m src [options]

Usage:
  python -m src                          # start with config.yaml
  python -m src --config /path/to.yaml   # custom config file
  python -m src --seed-goal "title" "description" --priority 1

The agent starts running immediately and loops forever.
Stop it with Ctrl+C.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="auto-mazur",
        description="Autonomous LLM agent loop",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--seed-goal",
        nargs=2,
        metavar=("TITLE", "DESCRIPTION"),
        help="Create an initial goal before starting the loop",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=1,
        help="Priority for --seed-goal (default: 1)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    log = logging.getLogger("auto-mazur")
    log.info("auto-mazur starting up")

    # Load config
    from .config import load_config
    try:
        config = load_config(args.config)
    except EnvironmentError as exc:
        log.error("Configuration error: %s", exc)
        sys.exit(1)

    log.info("Model: %s @ %s", config.model, config.api_base)
    log.info("Workspace: %s", config.workspace_root)
    log.info("Database: %s", config.db_path)

    # Audit logger — always created so LLM outputs and tool calls are recorded
    from .audit import AuditLogger
    audit = AuditLogger(store_root=config.store_root)
    log.info("Audit logs: %s/audit/", config.store_root)

    if config.gateway_enabled:
        log.info(
            "Gateway enabled — HTTP server will start on http://%s:%d",
            config.gateway_host, config.gateway_port,
        )

    # Build and start the main loop
    from .loop.main import MainLoop
    loop = MainLoop(config, audit=audit)
    loop.start()

    # Optionally seed an initial goal before the loop starts
    if args.seed_goal:
        title, description = args.seed_goal
        _seed_goal(loop, title, description, args.priority)

    log.info("Entering main loop — press Ctrl+C to stop")
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        loop.stop()


def _seed_goal(loop, title: str, description: str, priority: int) -> None:
    """Create an initial goal directly in the DB/workspace before the loop starts."""
    log = logging.getLogger("auto-mazur.seed")
    from .loop.actions import ActionExecutor
    from .models import ExecutiveAction

    executor = ActionExecutor(db=loop._db, workspace=loop._workspace, store=loop._store)
    action = ExecutiveAction(
        tool="create_goal",
        params={"title": title, "description": description, "priority": priority},
    )
    result = executor.execute(action)
    if result.error:
        log.error("Failed to seed goal: %s", result.error)
    else:
        log.info("Seeded goal: %r (priority %d)", title, priority)


if __name__ == "__main__":
    main()
