"""Unit tests for utils/cli.py."""

from __future__ import annotations

import logging

import pytest
import rich_click as click

from moment_to_action.utils.cli import GlobalData, ctx_get_seed, ctx_set_seed, get_global_data


def _make_ctx() -> click.Context:
    """Return a bare Click context backed by a throw-away command."""

    @click.command()
    def _dummy() -> None:
        pass

    return click.Context(_dummy)


@pytest.mark.unit
class TestCliUtils:
    """Tests for GlobalData, get_global_data, ctx_get_seed, ctx_set_seed."""

    def test_get_global_data_found(self) -> None:
        """Returns GlobalData when ctx.obj is a GlobalData instance."""
        ctx = _make_ctx()
        gd = GlobalData(log=logging.getLogger("test"))
        ctx.obj = gd
        assert get_global_data(ctx) is gd

    def test_get_global_data_not_found_raises(self) -> None:
        """Raises RuntimeError when no GlobalData exists in the context chain."""
        ctx = _make_ctx()
        with pytest.raises(RuntimeError, match="Global context data not found"):
            get_global_data(ctx)

    def test_ctx_set_seed_stores_value(self) -> None:
        """ctx_set_seed stores the provided seed in ctx.meta."""
        ctx = _make_ctx()
        ctx_set_seed(ctx, 12345)
        assert ctx_get_seed(ctx) == 12345

    def test_ctx_set_seed_none_generates_nonzero_seed(self) -> None:
        """ctx_set_seed with None generates a positive random seed."""
        ctx = _make_ctx()
        ctx_set_seed(ctx, None)
        seed = ctx_get_seed(ctx)
        assert isinstance(seed, int)
        assert seed > 0

    def test_ctx_get_seed_retrieves_stored_seed(self) -> None:
        """ctx_get_seed returns whatever was stored by ctx_set_seed."""
        ctx = _make_ctx()
        ctx_set_seed(ctx, 99)
        assert ctx_get_seed(ctx) == 99
