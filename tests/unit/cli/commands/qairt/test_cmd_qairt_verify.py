"""Unit tests for m2a qairt verify command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner, Result

from moment_to_action.config import AppConfig

if TYPE_CHECKING:
    from pathlib import Path


def _patched_pm(tmp_path: Path) -> MagicMock:
    mock_pm = MagicMock()
    mock_pm.data.data_dir = tmp_path
    mock_pm.app_config_file = tmp_path / "config.json"
    return mock_pm


def _invoke(args: list[str], tmp_path: Path, mgr: MagicMock) -> Result:
    from moment_to_action._cli import cli

    with patch("moment_to_action._cli.init_logging"):
        with patch("moment_to_action._cli.PathManager", return_value=_patched_pm(tmp_path)):
            with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                with patch(
                    "moment_to_action._cli.commands.cmd_qairt.cmd_verify.QairtSDKManager.from_app_config",
                    return_value=mgr,
                ):
                    return CliRunner().invoke(cli, ["qairt", "verify", *args])


@pytest.mark.unit
class TestQairtVerifyCommand:
    """Tests for the qairt verify subcommand."""

    def test_not_installed_shows_error(self, tmp_path: Path) -> None:
        """RuntimeError from verify is surfaced as a Click error."""
        mgr = MagicMock()
        mgr.verify.side_effect = RuntimeError("SDK not installed")
        result = _invoke([], tmp_path, mgr)
        assert result.exit_code != 0
        assert "not installed" in result.output

    def test_verify_success_exits_zero(self, tmp_path: Path) -> None:
        """No issues from verify exits 0."""
        mgr = MagicMock()
        mgr.verify.return_value = []
        result = _invoke([], tmp_path, mgr)
        assert result.exit_code == 0

    def test_verify_exits_one_when_issues(self, tmp_path: Path) -> None:
        """Issues returned by verify cause exit code 1."""
        mgr = MagicMock()
        mgr.verify.return_value = ["Missing system package: clang"]
        result = _invoke([], tmp_path, mgr)
        assert result.exit_code == 1

    def test_verify_called_with_stream_true(self, tmp_path: Path) -> None:
        """Verify calls mgr.verify(stream=True)."""
        mgr = MagicMock()
        mgr.verify.return_value = []
        _invoke([], tmp_path, mgr)
        mgr.verify.assert_called_once_with(stream=True)
