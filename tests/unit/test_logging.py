"""Unit tests for init_logging."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from moment_to_action._logging import init_logging


@pytest.mark.unit
class TestInitLogging:
    """Tests for init_logging."""

    def test_debug_level(self) -> None:
        """log_level='DEBUG' configures the root logger at DEBUG."""
        with patch("logging.basicConfig") as mock_cfg:
            init_logging(log_level="DEBUG")
            assert mock_cfg.call_args.kwargs["level"] == logging.DEBUG

    def test_info_level(self) -> None:
        """log_level='INFO' configures the root logger at INFO."""
        with patch("logging.basicConfig") as mock_cfg:
            init_logging(log_level="INFO")
            assert mock_cfg.call_args.kwargs["level"] == logging.INFO

    def test_warning_level(self) -> None:
        """log_level='WARNING' configures the root logger at WARNING."""
        with patch("logging.basicConfig") as mock_cfg:
            init_logging(log_level="WARNING")
            assert mock_cfg.call_args.kwargs["level"] == logging.WARNING

    def test_installs_rich_handler(self) -> None:
        """init_logging installs exactly one RichHandler."""
        from rich.logging import RichHandler

        with patch("logging.basicConfig") as mock_cfg:
            init_logging(log_level="INFO")
            handlers = mock_cfg.call_args.kwargs["handlers"]
            assert len(handlers) == 1
            assert isinstance(handlers[0], RichHandler)
