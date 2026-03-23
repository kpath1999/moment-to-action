"""CLI utilities."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    import logging

    import rich_click as click

_SEED_KEY = "moment_to_action.seed"
_BYTES_PER_UNIT = 1024


def format_size(size_bytes: int) -> str:
    """Format bytes as a human-readable size.

    Args:
        size_bytes: Number of bytes.

    Returns:
        Formatted size string (e.g., "13.0 MB", "1.5 GB").
    """
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < _BYTES_PER_UNIT:
            if unit == "B":
                return f"{int(value):.0f} {unit}"
            return f"{value:.1f} {unit}"
        value /= _BYTES_PER_UNIT
    return f"{value:.1f} TB"


@attrs.define
class GlobalData:
    """Click gobal context data."""

    log: logging.Logger
    """CLI logger."""


def get_global_data(ctx: click.Context) -> GlobalData:
    """Retrieve global context data from our click context."""
    if obj := ctx.find_object(GlobalData):
        return obj

    msg = "Global context data not found!"
    raise RuntimeError(msg)


def ctx_get_seed(ctx: click.Context) -> int:
    """Set seed in click context."""
    return ctx.meta[_SEED_KEY]


def ctx_set_seed(ctx: click.Context, seed: int | None) -> None:
    """Set seed in click context.

    If None is passed, a new one is generted.
    """
    ctx.meta[_SEED_KEY] = seed or secrets.randbits(64)
