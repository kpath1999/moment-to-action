"""Unit tests for m2a model remove command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

from pathlib import Path as _Path

import pytest
from click.testing import CliRunner, Result

from moment_to_action.config import AppConfig
from moment_to_action.models import ModelFormat, Variant
from moment_to_action.models._sources._vendored import VendoredSource


def _patched_pm(tmp_path: Path) -> MagicMock:
    pm = MagicMock()
    pm.app_config_file = tmp_path / "cfg.json"
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
        from moment_to_action.models import ModelID, ModelInfo, YOLOModel

        vendored_registry = {
            ModelID.YOLO_V8: ModelInfo(
                id=ModelID.YOLO_V8,
                model_class=YOLOModel,
                variants={
                    "default": Variant(
                        source=VendoredSource(format=ModelFormat.ONNX, path=_Path("yolo/m.onnx")),
                        backends={},
                    ),
                },
            )
        }
        mgr = MagicMock()
        with patch(
            "moment_to_action._cli.commands.cmd_model.cmd_remove.MODEL_REGISTRY",
            vendored_registry,
        ):
            result = _invoke(["yolo_v8", "--yes"], tmp_path, mgr)
        assert result.exit_code != 0
        assert "vendored" in result.output.lower()

    def test_removes_non_vendored_variant(self, tmp_path: Path) -> None:
        """Non-vendored variant is removed and bytes freed are printed."""
        mgr = MagicMock()
        mgr.remove_variant.return_value = 1024
        result = _invoke(["yolo_v8", "--variant", "qcs6490", "--yes"], tmp_path, mgr)
        assert result.exit_code == 0
        assert "1,024" in result.output

    def test_remove_variant_called(self, tmp_path: Path) -> None:
        """ModelManager.remove_variant is called with correct model_id and variant."""
        from moment_to_action.models import ModelID

        mgr = MagicMock()
        mgr.remove_variant.return_value = 512
        _invoke(["yolo_v8", "--variant", "qcs6490", "--yes"], tmp_path, mgr)
        mgr.remove_variant.assert_called_once_with(ModelID.YOLO_V8, "qcs6490")

    def test_confirm_prompt_shown_without_yes(self, tmp_path: Path) -> None:
        """Confirmation prompt is shown when --yes is omitted."""
        mgr = MagicMock()
        mgr.remove_variant.return_value = 0
        result = _invoke(["yolo_v8", "--variant", "qcs6490"], tmp_path, mgr)
        # Without input the prompt aborts
        assert result.exit_code != 0

    def test_remove_all_skips_vendored(self, tmp_path: Path) -> None:
        """--all does not attempt to remove vendored variants."""
        from moment_to_action.models import ModelID, ModelInfo, YOLOModel

        registry_with_vendored = {
            ModelID.YOLO_V8: ModelInfo(
                id=ModelID.YOLO_V8,
                model_class=YOLOModel,
                variants={
                    "default": Variant(
                        source=VendoredSource(format=ModelFormat.ONNX, path=_Path("yolo/m.onnx")),
                        backends={},
                    ),
                    "qcs6490": Variant(source=MagicMock(format=ModelFormat.DLC), backends={}),
                },
            )
        }
        mgr = MagicMock()
        mgr.is_available.return_value = False
        with patch(
            "moment_to_action._cli.commands.cmd_model.cmd_remove.MODEL_REGISTRY",
            registry_with_vendored,
        ):
            _invoke(["--all", "--yes"], tmp_path, mgr)
        # remove_variant should not be called for vendored 'default' variant
        for call in mgr.remove_variant.call_args_list:
            assert call[0][1] != "default"

    def test_remove_all_removes_cached_non_vendored(self, tmp_path: Path) -> None:
        """--all removes non-vendored variants that are available."""
        from moment_to_action.models import ModelID

        mgr = MagicMock()
        mgr.is_available.side_effect = lambda _, vkey: vkey == "qcs6490"
        mgr.remove_variant.return_value = 2048
        result = _invoke(["--all", "--yes"], tmp_path, mgr)
        assert result.exit_code == 0
        # Multiple models register a "qcs6490" variant; YOLO's must be among those removed.
        mgr.remove_variant.assert_any_call(ModelID.YOLO_V8, "qcs6490")

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
        mgr = MagicMock()
        mgr.is_available.return_value = False
        result = _invoke(["--all"], tmp_path, mgr)
        # Without providing input, the prompt aborts
        assert result.exit_code != 0
        # Confirmation text should be in output
        assert "Remove all" in result.output or "abort" in result.output.lower()
