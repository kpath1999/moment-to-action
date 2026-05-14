"""Unit tests for m2a qairt install command."""

from __future__ import annotations

import json
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
                    "moment_to_action._cli.commands.cmd_qairt.cmd_install.QairtSDKManager.from_app_config",
                    return_value=mgr,
                ):
                    return CliRunner().invoke(cli, ["qairt", "install", *args])


@pytest.mark.unit
class TestQairtInstallCommand:
    """Tests for the qairt install subcommand."""

    def test_missing_deps_aborts(self, tmp_path: Path) -> None:
        """Missing system deps aborts before attempting install."""
        mgr = MagicMock()
        mgr.check_system_deps.return_value = ["libgl1", "clang"]
        result = _invoke([], tmp_path, mgr)
        assert result.exit_code != 0
        mgr.install.assert_not_called()

    def test_install_failure_raises_click_exception(self, tmp_path: Path) -> None:
        """RuntimeError from install is surfaced as a Click error."""
        mgr = MagicMock()
        mgr.check_system_deps.return_value = []
        mgr.install.side_effect = RuntimeError("fetch failed (exit 1)")
        result = _invoke([], tmp_path, mgr)
        assert result.exit_code != 0
        assert "fetch failed" in result.output

    def test_happy_path_saves_config_and_calls_verify(self, tmp_path: Path) -> None:
        """Successful install saves config and runs verify."""
        sdk_path = tmp_path / "qairt" / "2.45.0.24"
        sdk_path.mkdir(parents=True)
        mgr = MagicMock()
        mgr.check_system_deps.return_value = []
        mgr.install.return_value = sdk_path
        mgr.installed_version = "2.45.0.24"
        mgr.verify.return_value = []
        result = _invoke([], tmp_path, mgr)
        assert result.exit_code == 0
        mgr.install.assert_called_once()
        mgr.verify.assert_called_once_with(stream=True)

    def test_verify_warnings_are_logged(self, tmp_path: Path) -> None:
        """Verify warnings are logged after a successful install."""
        sdk_path = tmp_path / "qairt" / "2.45.0.24"
        sdk_path.mkdir(parents=True)
        mgr = MagicMock()
        mgr.check_system_deps.return_value = []
        mgr.install.return_value = sdk_path
        mgr.installed_version = "2.45.0.24"
        mgr.verify.return_value = ["Missing system package: clang"]
        result = _invoke([], tmp_path, mgr)
        assert result.exit_code == 0

    def test_json_output_contains_path_and_version(self, tmp_path: Path) -> None:
        """--json flag outputs valid JSON with path and version keys."""
        sdk_path = tmp_path / "qairt" / "2.45.0.24"
        sdk_path.mkdir(parents=True)
        mgr = MagicMock()
        mgr.check_system_deps.return_value = []
        mgr.install.return_value = sdk_path
        mgr.installed_version = "2.45.0.24"
        mgr.verify.return_value = []
        result = _invoke(["--json"], tmp_path, mgr)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "path" in data
        assert "version" in data
        assert data["version"] == "2.45.0.24"
