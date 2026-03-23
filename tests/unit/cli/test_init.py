"""Unit tests for _cli/__init__.py (top-level cli group) and __main__.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


@pytest.mark.unit
class TestMainModule:
    """Tests for __main__.py entrypoint."""

    def test_main_module_imports_cli(self) -> None:
        """Importing __main__ exposes the cli callable (covers the import line)."""
        import importlib

        mod = importlib.import_module("moment_to_action.__main__")
        assert callable(mod.cli)


@pytest.mark.unit
class TestCliCommand:
    """Tests for the top-level cli Click group."""

    def _mock_backend(self) -> MagicMock:
        sample = MagicMock()
        sample.power_mw = 100
        sample.utilization_pct = 10
        pwr_mon = MagicMock()
        pwr_mon.sample.return_value = sample
        backend = MagicMock()
        backend.power_monitor = pwr_mon
        return backend

    def test_cli_help_exits_cleanly(self) -> None:
        """--help prints usage and exits 0."""
        from moment_to_action._cli import cli

        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "read-power" in result.output or "rdpwr" in result.output

    def test_cli_verbose_flag_calls_init_logging_with_true(self) -> None:
        """--verbose causes init_logging(verbose=True)."""
        from moment_to_action._cli import cli

        with patch("moment_to_action._cli.init_logging") as mock_init:
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=self._mock_backend(),
            ):
                result = CliRunner().invoke(cli, ["--verbose", "read-power", "CPU"])
        assert result.exit_code == 0
        mock_init.assert_called_once_with(verbose=True)

    def test_cli_default_verbose_false(self) -> None:
        """Without --verbose, init_logging is called with verbose=False."""
        from moment_to_action._cli import cli

        with patch("moment_to_action._cli.init_logging") as mock_init:
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=self._mock_backend(),
            ):
                result = CliRunner().invoke(cli, ["read-power", "CPU"])
        assert result.exit_code == 0
        mock_init.assert_called_once_with(verbose=False)

    def test_cli_seed_option_accepted(self) -> None:
        """--seed with a hex value is accepted without error."""
        from moment_to_action._cli import cli

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=self._mock_backend(),
            ):
                result = CliRunner().invoke(cli, ["--seed", "0x2a", "read-power", "CPU"])
        assert result.exit_code == 0
