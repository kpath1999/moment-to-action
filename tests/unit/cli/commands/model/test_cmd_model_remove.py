"""Unit tests for m2a model remove command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from moment_to_action.config import AppConfig


def _patched_pm(tmp_path: Path, freed: int = 0) -> MagicMock:
    pm = MagicMock()
    pm.app_config_file = tmp_path / "cfg.json"
    pm.cache.models.remove_variant.return_value = freed
    return pm


def _invoke(
    args: list[str],
    tmp_path: Path,
    mock_mgr: MagicMock,
    patched_pm: MagicMock | None = None,
) -> Result:
    from moment_to_action._cli import cli

    pm = patched_pm or _patched_pm(tmp_path)
    with patch("moment_to_action._cli.init_logging"):
        with patch("moment_to_action._cli.PathManager", return_value=pm):
            with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                with patch(
                    "moment_to_action._cli.commands.cmd_model.cmd_remove.ModelManager",
                    return_value=mock_mgr,
                ):
                    return CliRunner().invoke(cli, ["model", "remove", *args])


@pytest.mark.unit
class TestModelRemoveCommand:
    """Tests for m2a model remove."""

    def test_vendored_variant_rejected(self, tmp_path: Path) -> None:
        """Attempting to remove a vendored variant exits non-zero."""
        mgr = MagicMock()
        # Default variant of yolo_v8 is VendoredSource
        result = _invoke(["yolo_v8", "--yes"], tmp_path, mgr)
        assert result.exit_code != 0
        assert "vendored" in result.output.lower()

    def test_removes_non_vendored_variant(self, tmp_path: Path) -> None:
        """Non-vendored variant is removed and bytes freed are printed."""
        pm = _patched_pm(tmp_path, freed=1024)
        mgr = MagicMock()
        result = _invoke(["yolo_v8", "--variant", "qcs6490", "--yes"], tmp_path, mgr, pm)
        assert result.exit_code == 0
        assert "1,024" in result.output

    def test_remove_variant_called(self, tmp_path: Path) -> None:
        """remove_variant is called with correct model_id and variant."""
        pm = _patched_pm(tmp_path, freed=512)
        mgr = MagicMock()
        _invoke(["yolo_v8", "--variant", "qcs6490", "--yes"], tmp_path, mgr, pm)
        pm.cache.models.remove_variant.assert_called_once_with("yolo_v8", "qcs6490")

    def test_confirm_prompt_shown_without_yes(self, tmp_path: Path) -> None:
        """Confirmation prompt is shown when --yes is omitted."""
        pm = _patched_pm(tmp_path, freed=0)
        mgr = MagicMock()
        result = _invoke(["yolo_v8", "--variant", "qcs6490"], tmp_path, mgr, pm)
        # Without input the prompt aborts
        assert result.exit_code != 0

    def test_remove_all_skips_vendored(self, tmp_path: Path) -> None:
        """--all does not attempt to remove vendored variants."""
        pm = _patched_pm(tmp_path, freed=0)
        mgr = MagicMock()
        mgr.is_available.return_value = False
        _invoke(["--all", "--yes"], tmp_path, mgr, pm)
        # remove_variant should not be called for vendored 'default' variant
        for call in pm.cache.models.remove_variant.call_args_list:
            assert call[0][1] != "default"

    def test_remove_all_removes_cached_non_vendored(self, tmp_path: Path) -> None:
        """--all removes non-vendored variants that are available."""
        pm = _patched_pm(tmp_path, freed=2048)
        mgr = MagicMock()
        mgr.is_available.side_effect = lambda _, vkey: vkey == "qcs6490"
        result = _invoke(["--all", "--yes"], tmp_path, mgr, pm)
        assert result.exit_code == 0
        pm.cache.models.remove_variant.assert_called_with("yolo_v8", "qcs6490")

    def test_no_model_id_without_all_errors(self, tmp_path: Path) -> None:
        """Missing MODEL_ID without --all exits non-zero."""
        mgr = MagicMock()
        result = _invoke([], tmp_path, mgr)
        assert result.exit_code != 0

    def test_unknown_variant_errors(self, tmp_path: Path) -> None:
        """Unknown variant name exits non-zero."""
        mgr = MagicMock()
        result = _invoke(["yolo_v8", "--variant", "nonexistent", "--yes"], tmp_path, mgr)
        assert result.exit_code != 0

    def test_remove_all_shows_confirmation_prompt(self, tmp_path: Path) -> None:
        """--all without --yes shows confirmation prompt."""
        pm = _patched_pm(tmp_path, freed=0)
        mgr = MagicMock()
        mgr.is_available.return_value = False
        result = _invoke(["--all"], tmp_path, mgr, pm)
        # Without providing input, the prompt aborts
        assert result.exit_code != 0
        # Confirmation text should be in output
        assert "Remove all" in result.output or "abort" in result.output.lower()
