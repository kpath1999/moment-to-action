"""Unit tests for m2a model inspect command."""

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
                    "moment_to_action._cli.commands.cmd_model.cmd_inspect.ModelManager",
                    return_value=mock_mgr,
                ):
                    return CliRunner().invoke(cli, ["model", "inspect", *args])


@pytest.mark.unit
class TestModelInspectCommand:
    """Tests for m2a model inspect."""

    def test_exits_zero_not_cached(self, tmp_path: Path) -> None:
        """Exits 0 when model is not cached."""
        mgr = MagicMock()
        mgr.is_available.return_value = False
        result = _invoke(["yolo_v8"], tmp_path, mgr)
        assert result.exit_code == 0

    def test_shows_not_cached_when_unavailable(self, tmp_path: Path) -> None:
        """Shows 'not cached' label when model is unavailable."""
        mgr = MagicMock()
        mgr.is_available.return_value = False
        result = _invoke(["yolo_v8"], tmp_path, mgr)
        assert "not cached" in result.output

    def test_shows_source_type_always(self, tmp_path: Path) -> None:
        """Source type is always shown."""
        mgr = MagicMock()
        mgr.is_available.return_value = False
        result = _invoke(["yolo_v8"], tmp_path, mgr)
        assert result.exit_code == 0
        assert "Source type" in result.output

    def test_shows_format_always(self, tmp_path: Path) -> None:
        """Format is always shown."""
        mgr = MagicMock()
        mgr.is_available.return_value = False
        result = _invoke(["yolo_v8"], tmp_path, mgr)
        assert "Format" in result.output

    def test_shows_path_when_cached_file(self, tmp_path: Path) -> None:
        """Path and SHA-256 shown when model is a single file."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake_model_content")
        mgr = MagicMock()
        mgr.is_available.return_value = True
        mgr.get_path.return_value = model_file
        result = _invoke(["yolo_v8"], tmp_path, mgr)
        assert result.exit_code == 0
        assert "Path" in result.output
        assert "SHA-256" in result.output

    def test_shows_directory_label_when_cached_dir(self, tmp_path: Path) -> None:
        """Shows 'directory' label when model path is a directory."""
        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()
        (model_dir / "file1.dlc").write_bytes(b"a")
        mgr = MagicMock()
        mgr.is_available.return_value = True
        mgr.get_path.return_value = model_dir
        result = _invoke(["yolo_v8"], tmp_path, mgr)
        assert result.exit_code == 0
        assert "directory" in result.output

    def test_variant_option_passed_through(self, tmp_path: Path) -> None:
        """--variant is forwarded to is_available and get_path."""
        mgr = MagicMock()
        mgr.is_available.return_value = False
        _invoke(["yolo_v8", "--variant", "qcs6490"], tmp_path, mgr)
        mgr.is_available.assert_called_once_with(ModelID.YOLO_V8, "qcs6490")
