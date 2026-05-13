"""Unit tests for m2a config CLI command."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from click.testing import CliRunner, Result

if TYPE_CHECKING:
    from moment_to_action.paths import PathManager


@pytest.mark.unit
class TestConfigCommand:
    """Tests for the `m2a config` command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Return a Click test runner."""
        return CliRunner()

    @pytest.fixture(autouse=True)
    def _patch_path_manager(
        self, path_manager: PathManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Patch PathManager in CLI init to use tmp_path-rooted paths."""
        monkeypatch.setattr(
            "moment_to_action._cli.PathManager",
            lambda: path_manager,
        )

    def _invoke(self, runner: CliRunner, args: list[str]) -> Result:
        from moment_to_action._cli import cli

        with patch("moment_to_action._cli.init_logging"):
            return runner.invoke(cli, ["config", *args])

    def test_no_args_prints_rich_table(self, runner: CliRunner) -> None:
        """No arguments prints full config as a rich table."""
        result = self._invoke(runner, [])
        assert result.exit_code == 0
        assert "max_workers" in result.output
        assert "log_level" in result.output

    def test_no_args_json_flag(self, runner: CliRunner) -> None:
        """No arguments with --json prints full config as JSON."""
        result = self._invoke(runner, ["--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "max_workers" in data
        assert "log_level" in data

    def test_get_key(self, runner: CliRunner) -> None:
        """KEY only prints the current value."""
        result = self._invoke(runner, ["log_level"])
        assert result.exit_code == 0
        assert "INFO" in result.output

    def test_set_key(self, runner: CliRunner) -> None:
        """KEY VALUE sets the value and prints confirmation."""
        result = self._invoke(runner, ["max_workers", "4"])
        assert result.exit_code == 0
        assert "max_workers = 4" in result.output

    def test_set_persists_value(self, runner: CliRunner) -> None:
        """Setting a value persists so a subsequent get returns the new value."""
        self._invoke(runner, ["max_workers", "6"])
        result = self._invoke(runner, ["max_workers"])
        assert result.exit_code == 0
        assert "6" in result.output

    def test_unknown_key_errors(self, runner: CliRunner) -> None:
        """Unknown key returns a non-zero exit code."""
        result = self._invoke(runner, ["nonexistent_key"])
        assert result.exit_code != 0

    def test_invalid_value_errors(self, runner: CliRunner) -> None:
        """Passing a value that fails pydantic validation returns a non-zero exit code."""
        result = self._invoke(runner, ["max_workers", "not_a_number"])
        assert result.exit_code != 0

    def test_invalid_log_level_errors(self, runner: CliRunner) -> None:
        """Passing an invalid log level returns a non-zero exit code."""
        result = self._invoke(runner, ["log_level", "TRACE"])
        assert result.exit_code != 0
