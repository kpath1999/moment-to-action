"""Unit tests for m2a qairt clean command."""

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


def _invoke(args: list[str], tmp_path: Path, mgr: MagicMock, stdin: str | None = None) -> Result:
    from moment_to_action._cli import cli

    with patch("moment_to_action._cli.init_logging"):
        with patch("moment_to_action._cli.PathManager", return_value=_patched_pm(tmp_path)):
            with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                with patch(
                    "moment_to_action._cli.commands.cmd_qairt.cmd_clean.QairtSDKManager.from_app_config",
                    return_value=mgr,
                ):
                    return CliRunner().invoke(cli, ["qairt", "clean", *args], input=stdin)


@pytest.mark.unit
class TestQairtCleanCommand:
    """Tests for the qairt clean subcommand."""

    def test_not_installed_shows_error(self, tmp_path: Path) -> None:
        """RuntimeError from clean is surfaced as a Click error."""
        mgr = MagicMock()
        mgr.clean.side_effect = RuntimeError("SDK not installed via m2a")
        result = _invoke(["--force"], tmp_path, mgr)
        assert result.exit_code != 0
        assert "not installed" in result.output

    def test_force_skips_confirmation(self, tmp_path: Path) -> None:
        """--force skips the confirmation prompt and proceeds."""
        removed = tmp_path / "2.45.0.24"
        mgr = MagicMock()
        mgr.clean.return_value = removed
        result = _invoke(["--force"], tmp_path, mgr)
        assert result.exit_code == 0
        mgr.clean.assert_called_once()

    def test_confirm_yes_proceeds(self, tmp_path: Path) -> None:
        """Confirmation 'y' proceeds with removal."""
        removed = tmp_path / "2.45.0.24"
        mgr = MagicMock()
        mgr.clean.return_value = removed
        result = _invoke([], tmp_path, mgr, stdin="y\n")
        assert result.exit_code == 0
        mgr.clean.assert_called_once()

    def test_confirm_no_cancels(self, tmp_path: Path) -> None:
        """Confirmation 'n' cancels without removing."""
        mgr = MagicMock()
        result = _invoke([], tmp_path, mgr, stdin="n\n")
        assert result.exit_code == 0
        assert "cancelled" in result.output.lower()
        mgr.clean.assert_not_called()

    def test_json_skips_confirmation_and_outputs_json(self, tmp_path: Path) -> None:
        """--json skips confirmation and outputs valid JSON."""
        removed = tmp_path / "2.45.0.24"
        mgr = MagicMock()
        mgr.clean.return_value = removed
        result = _invoke(["--json"], tmp_path, mgr)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "removed" in data
        assert data["removed"] == str(removed)
        mgr.clean.assert_called_once()

    def test_eof_in_non_interactive_proceeds(self, tmp_path: Path) -> None:
        """EOFError in non-interactive mode is treated as confirmation and proceeds."""
        removed = tmp_path / "2.45.0.24"
        mgr = MagicMock()
        mgr.clean.return_value = removed
        result = _invoke([], tmp_path, mgr, stdin="")
        assert result.exit_code == 0
        mgr.clean.assert_called_once()

    def test_config_cleared_after_clean(self, tmp_path: Path) -> None:
        """After clean, qairt_sdk_path is set to None in saved config."""
        removed = tmp_path / "2.45.0.24"
        sdk_path = removed
        config = AppConfig(qairt_sdk_path=sdk_path)
        mgr = MagicMock()
        mgr.clean.return_value = removed

        from moment_to_action._cli import cli

        saved_configs: list[AppConfig] = []

        def _save(cfg: AppConfig, _path: object) -> None:
            saved_configs.append(cfg)

        with patch("moment_to_action._cli.init_logging"):
            with patch("moment_to_action._cli.PathManager", return_value=_patched_pm(tmp_path)):
                with patch("moment_to_action._cli.load_config", return_value=config):
                    with patch(
                        "moment_to_action._cli.commands.cmd_qairt.cmd_clean.QairtSDKManager.from_app_config",
                        return_value=mgr,
                    ):
                        with patch(
                            "moment_to_action._cli.commands.cmd_qairt.cmd_clean.save_config",
                            side_effect=_save,
                        ):
                            CliRunner().invoke(cli, ["qairt", "clean", "--force"])

        assert saved_configs, "save_config was not called"
        assert saved_configs[-1].qairt_sdk_path is None
