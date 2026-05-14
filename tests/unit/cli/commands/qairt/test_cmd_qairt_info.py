"""Unit tests for m2a qairt info command."""

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
                    "moment_to_action._cli.commands.cmd_qairt.cmd_info.QairtSDKManager.from_app_config",
                    return_value=mgr,
                ):
                    return CliRunner().invoke(cli, ["qairt", "info", *args])


def _make_mgr(
    *,
    configured_version: str = "2.45.0",
    installed_version: str | None = None,
    path: Path | None = None,
    available: bool = False,
) -> MagicMock:
    mgr = MagicMock()
    mgr.configured_version = configured_version
    mgr.installed_version = installed_version
    mgr.path = path
    mgr.is_available = available
    return mgr


@pytest.mark.unit
class TestQairtInfoCommand:
    """Tests for the qairt info subcommand."""

    def test_not_installed_exits_zero(self, tmp_path: Path) -> None:
        """Command exits 0 even when SDK is not installed."""
        mgr = _make_mgr()
        result = _invoke([], tmp_path, mgr)
        assert result.exit_code == 0

    def test_not_installed_shows_not_installed(self, tmp_path: Path) -> None:
        """Output contains 'not installed' indicator when SDK is absent."""
        mgr = _make_mgr()
        result = _invoke([], tmp_path, mgr)
        assert result.exit_code == 0
        assert "not installed" in result.output.lower()

    def test_installed_shows_version_and_path(self, tmp_path: Path) -> None:
        """Output contains version and path indicator when SDK is installed."""
        sdk = tmp_path / "2.45.0.24"
        mgr = _make_mgr(
            installed_version="2.45.0.24",
            path=sdk,
            available=True,
        )
        result = _invoke([], tmp_path, mgr)
        assert result.exit_code == 0
        assert "2.45.0.24" in result.output

    def test_json_output_valid_json(self, tmp_path: Path) -> None:
        """--json flag produces valid JSON."""
        mgr = _make_mgr()
        result = _invoke(["--json"], tmp_path, mgr)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, dict)

    def test_json_output_has_expected_keys(self, tmp_path: Path) -> None:
        """JSON output contains all expected keys."""
        sdk = tmp_path / "2.45.0.24"
        mgr = _make_mgr(
            configured_version="2.45.0",
            installed_version="2.45.0.24",
            path=sdk,
            available=True,
        )
        result = _invoke(["--json"], tmp_path, mgr)
        data = json.loads(result.output)
        assert "configured_version" in data
        assert "installed_version" in data
        assert "path" in data
        assert "available" in data

    def test_json_not_installed_path_is_null(self, tmp_path: Path) -> None:
        """JSON output has null path and available=false when SDK not installed."""
        mgr = _make_mgr()
        result = _invoke(["--json"], tmp_path, mgr)
        data = json.loads(result.output)
        assert data["path"] is None
        assert data["available"] is False

    def test_json_installed_path_is_string(self, tmp_path: Path) -> None:
        """JSON output has string path and available=true when SDK is installed."""
        sdk = tmp_path / "2.45.0.24"
        mgr = _make_mgr(path=sdk, available=True, installed_version="2.45.0.24")
        result = _invoke(["--json"], tmp_path, mgr)
        data = json.loads(result.output)
        assert data["path"] == str(sdk)
        assert data["available"] is True
