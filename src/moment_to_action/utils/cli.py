"""CLI utilities."""

import logging
import secrets

import attrs
import rich_click as click

_SEED_KEY = "moment_to_action.seed"


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
