"""Unit tests for m2a model convert command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner, Result

from moment_to_action.config import AppConfig


def _patched_pm(tmp_path: Path) -> MagicMock:
    pm = MagicMock()
    pm.app_config_file = tmp_path / "cfg.json"
    return pm


def _invoke(
    args: list[str],
    tmp_path: Path,
    mock_mgr: MagicMock,
    mock_qairt_mgr: MagicMock,
    mock_backend_cls: MagicMock | None = None,
) -> Result:
    from moment_to_action._cli import cli

    backend_cls = mock_backend_cls or MagicMock()
    with patch("moment_to_action._cli.init_logging"):
        with patch("moment_to_action._cli.PathManager", return_value=_patched_pm(tmp_path)):
            with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                with patch(
                    "moment_to_action._cli.commands.cmd_model.cmd_convert.ModelManager",
                    return_value=mock_mgr,
                ):
                    with patch(
                        "moment_to_action._cli.commands.cmd_model.cmd_convert.QairtSDKManager.from_app_config",
                        return_value=mock_qairt_mgr,
                    ):
                        with patch(
                            "moment_to_action._cli.commands.cmd_model.cmd_convert.ComputeBackend",
                            return_value=backend_cls,
                        ):
                            return CliRunner().invoke(cli, ["model", "convert", *args])


def _make_qairt_mgr(*, available: bool = True) -> MagicMock:
    mgr = MagicMock()
    mgr.is_available = available
    mgr.convert.return_value = None
    return mgr


def _make_model_mgr(tmp_path: Path) -> MagicMock:
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"onnx")
    mgr = MagicMock()
    from moment_to_action.models.image.detection._base import ImageDetectionModel

    mock_model = MagicMock(spec=ImageDetectionModel)
    mock_model.path = model_file
    mock_model.prepare.return_value = np.zeros((1, 3, 640, 640), dtype=np.float32)
    mock_model.run.return_value = [np.zeros((1, 10, 4)), np.zeros((1, 10)), np.zeros((1, 10))]
    mock_model.prepare_for_conversion.return_value = model_file
    mgr.get_model.return_value = mock_model
    return mgr


@pytest.mark.unit
class TestModelConvertCommand:
    """Tests for m2a model convert."""

    def test_happy_path(self, tmp_path: Path) -> None:
        """Successful conversion exits 0 and prints output dir."""
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2

        cv2.imwrite(str(calib_dir / "img.jpg"), img)

        output_dir = tmp_path / "out"
        qairt_mgr = _make_qairt_mgr(available=True)
        model_mgr = _make_model_mgr(tmp_path)

        result = _invoke(
            [
                "yolo_v8",
                "-o",
                str(output_dir),
                "--calibration-dir",
                str(calib_dir),
            ],
            tmp_path,
            model_mgr,
            qairt_mgr,
        )
        assert result.exit_code == 0
        assert "Converted:" in result.output

    def test_sdk_not_available_errors(self, tmp_path: Path) -> None:
        """Exits non-zero when QAIRT SDK is not installed."""
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        output_dir = tmp_path / "out"
        qairt_mgr = _make_qairt_mgr(available=False)
        model_mgr = MagicMock()

        result = _invoke(
            [
                "yolo_v8",
                "-o",
                str(output_dir),
                "--calibration-dir",
                str(calib_dir),
            ],
            tmp_path,
            model_mgr,
            qairt_mgr,
        )
        assert result.exit_code != 0
        assert "QAIRT SDK not installed" in result.output

    def test_empty_calibration_dir_errors(self, tmp_path: Path) -> None:
        """Exits non-zero when calibration dir has no images."""
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        output_dir = tmp_path / "out"
        qairt_mgr = _make_qairt_mgr(available=True)
        model_mgr = _make_model_mgr(tmp_path)

        result = _invoke(
            [
                "yolo_v8",
                "-o",
                str(output_dir),
                "--calibration-dir",
                str(calib_dir),
            ],
            tmp_path,
            model_mgr,
            qairt_mgr,
        )
        assert result.exit_code != 0

    def test_reference_outputs_written(self, tmp_path: Path) -> None:
        """Reference outputs directory is created with inputs.npy and outputs_*.npy."""
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2

        cv2.imwrite(str(calib_dir / "img.jpg"), img)

        output_dir = tmp_path / "out"
        qairt_mgr = _make_qairt_mgr(available=True)
        model_mgr = _make_model_mgr(tmp_path)

        _invoke(
            [
                "yolo_v8",
                "-o",
                str(output_dir),
                "--calibration-dir",
                str(calib_dir),
            ],
            tmp_path,
            model_mgr,
            qairt_mgr,
        )
        ref_dir = output_dir / "reference_outputs"
        assert ref_dir.exists()
        assert (ref_dir / "inputs.npy").exists()
        assert (ref_dir / "outputs_0.npy").exists()

    def test_convert_called_with_dlc_path(self, tmp_path: Path) -> None:
        """QairtSDKManager.convert is called with the dlc output path."""
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2

        cv2.imwrite(str(calib_dir / "img.jpg"), img)

        output_dir = tmp_path / "out"
        qairt_mgr = _make_qairt_mgr(available=True)
        model_mgr = _make_model_mgr(tmp_path)

        _invoke(
            [
                "yolo_v8",
                "-o",
                str(output_dir),
                "--calibration-dir",
                str(calib_dir),
            ],
            tmp_path,
            model_mgr,
            qairt_mgr,
        )
        assert qairt_mgr.convert.called
        call_kwargs = qairt_mgr.convert.call_args
        # arg[0] = model.path, arg[1] = dlc output path
        dlc_path = call_kwargs[0][1]
        assert str(dlc_path).endswith("model.dlc")

    def test_prepare_for_conversion_result_passed_to_qairt(self, tmp_path: Path) -> None:
        """qairt.convert() receives the path returned by prepare_for_conversion()."""
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2

        cv2.imwrite(str(calib_dir / "img.jpg"), img)

        surgery_path = tmp_path / "surgery.onnx"
        surgery_path.write_bytes(b"onnx-surgery")

        output_dir = tmp_path / "out"
        qairt_mgr = _make_qairt_mgr(available=True)
        model_mgr = _make_model_mgr(tmp_path)
        model_mgr.get_model.return_value.prepare_for_conversion.return_value = surgery_path

        _invoke(
            [
                "yolo_v8",
                "-o",
                str(output_dir),
                "--calibration-dir",
                str(calib_dir),
            ],
            tmp_path,
            model_mgr,
            qairt_mgr,
        )
        call_args = qairt_mgr.convert.call_args[0]
        assert call_args[0] == surgery_path

    def test_non_image_model_errors(self, tmp_path: Path) -> None:
        """Exits non-zero when model is not an ImageModel."""
        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        output_dir = tmp_path / "out"
        qairt_mgr = _make_qairt_mgr(available=True)
        # Plain MagicMock is not an ImageModel
        non_image_mgr = MagicMock()
        non_image_mgr.get_model.return_value = MagicMock()

        result = _invoke(
            [
                "yolo_v8",
                "-o",
                str(output_dir),
                "--calibration-dir",
                str(calib_dir),
            ],
            tmp_path,
            non_image_mgr,
            qairt_mgr,
        )
        assert result.exit_code != 0
        assert "image detection model" in result.output
