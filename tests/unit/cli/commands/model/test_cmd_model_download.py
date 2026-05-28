"""Unit tests for m2a model download command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from moment_to_action.config import AppConfig
from moment_to_action.models import ModelID


def _invoke(args: list[str], tmp_path: Path, mock_mgr: MagicMock) -> Result:
    from moment_to_action._cli import cli

    with patch("moment_to_action._cli.init_logging"):
        with patch(
            "moment_to_action._cli.PathManager",
            return_value=MagicMock(app_config_file=tmp_path / "cfg.json"),
        ):
            with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                with patch(
                    "moment_to_action._cli.commands.cmd_model.cmd_download.ModelManager",
                    return_value=mock_mgr,
                ):
                    return CliRunner().invoke(cli, ["model", "download", *args])


@pytest.mark.unit
class TestModelDownloadCommand:
    """Tests for m2a model download."""

    def test_exits_zero_on_success(self, tmp_path: Path) -> None:
        """Successful download exits 0."""
        mgr = MagicMock()
        mgr.get_path.return_value = tmp_path / "model.onnx"
        result = _invoke(["yolo_v8"], tmp_path, mgr)
        assert result.exit_code == 0

    def test_prints_path_on_success(self, tmp_path: Path) -> None:
        """Output contains 'Downloaded:' with the path."""
        expected = tmp_path / "model.onnx"
        mgr = MagicMock()
        mgr.get_path.return_value = expected
        result = _invoke(["yolo_v8"], tmp_path, mgr)
        assert "Downloaded:" in result.output
        assert str(expected) in result.output

    def test_calls_get_path_with_default_variant(self, tmp_path: Path) -> None:
        """get_path called with default variant when --variant omitted."""
        from moment_to_action.models import DEFAULT_VARIANT_KEY

        mgr = MagicMock()
        mgr.get_path.return_value = tmp_path / "model.onnx"
        _invoke(["yolo_v8"], tmp_path, mgr)
        mgr.get_path.assert_called_once_with(ModelID.YOLO_V8, DEFAULT_VARIANT_KEY)

    def test_calls_get_path_with_specified_variant(self, tmp_path: Path) -> None:
        """get_path called with the specified --variant."""
        mgr = MagicMock()
        mgr.get_path.return_value = tmp_path / "model.dlc"
        _invoke(["yolo_v8", "--variant", "qcs6490"], tmp_path, mgr)
        mgr.get_path.assert_called_once_with(ModelID.YOLO_V8, "qcs6490")

    def test_invalid_model_id_exits_nonzero(self, tmp_path: Path) -> None:
        """Unknown model ID causes non-zero exit."""
        mgr = MagicMock()
        result = _invoke(["nonexistent_model"], tmp_path, mgr)
        assert result.exit_code != 0
