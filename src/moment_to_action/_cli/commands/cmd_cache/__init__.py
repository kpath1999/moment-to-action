"""Cache management subcommand group."""

from __future__ import annotations

from pathlib import Path

from moment_to_action._cli._auto_group import auto_group


@auto_group(cmd_path=Path(__file__).parent)
def cache() -> None:
    """Manage model cache (inspect, clear)."""
