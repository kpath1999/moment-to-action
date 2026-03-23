"""Unit tests for init_logging."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from moment_to_action._logging import init_logging


@pytest.mark.unit
class TestInitLogging:
    """Tests for init_logging."""

    def test_verbose_sets_debug_level(self) -> None:
        """verbose=True configures the root logger at DEBUG."""
        with patch("logging.basicConfig") as mock_cfg:
            init_logging(verbose=True)
            assert mock_cfg.call_args.kwargs["level"] == logging.DEBUG

    def test_non_verbose_sets_info_level(self) -> None:
        """verbose=False configures the root logger at INFO."""
        with patch("logging.basicConfig") as mock_cfg:
            init_logging(verbose=False)
            assert mock_cfg.call_args.kwargs["level"] == logging.INFO

    def test_installs_rich_handler(self) -> None:
        """init_logging installs exactly one RichHandler."""
        from rich.logging import RichHandler

        with patch("logging.basicConfig") as mock_cfg:
            init_logging(verbose=False)
            handlers = mock_cfg.call_args.kwargs["handlers"]
            assert len(handlers) == 1
            assert isinstance(handlers[0], RichHandler)
