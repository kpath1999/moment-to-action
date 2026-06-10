"""Unit tests for m2a model run command."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner, Result

from moment_to_action.config import AppConfig
from moment_to_action.models.image._base import ImageModel
from moment_to_action.models.image.classification._base import ImageClassificationModel
from moment_to_action.models.image.classification._types import Classification
from moment_to_action.models.image.detection._types import BoundingBox, Detection


def _patched_pm(tmp_path: Path) -> MagicMock:
    pm = MagicMock()
    pm.app_config_file = tmp_path / "cfg.json"
    return pm


def _make_detection(label: str = "person", conf: float = 0.9) -> Detection:
    return Detection(label=label, confidence=conf, bbox=BoundingBox(10, 20, 100, 200))


def _make_mock_model(detections: list[Detection] | None = None) -> MagicMock:
    """Build an ImageModel mock with canned inference output."""
    mock_model = MagicMock(spec=ImageModel)
    mock_model.prepare.return_value = np.zeros((1, 3, 640, 640), dtype=np.float32)
    mock_model.run.return_value = [np.zeros((1, 5, 4))]
    mock_model.post_proc.return_value = detections if detections is not None else []
    return mock_model


def _make_classification(label: str = "tench", conf: float = 0.95) -> Classification:
    """Return a single Classification for use in tests."""
    return Classification(label=label, confidence=conf, class_id=0)


def _make_mock_classification_model(
    classifications: list[Classification] | None = None,
) -> MagicMock:
    """Build an ImageClassificationModel mock with canned inference output."""
    mock_model = MagicMock(spec=ImageClassificationModel)
    mock_model.prepare.return_value = np.zeros((1, 3, 224, 224), dtype=np.float32)
    mock_model.run.return_value = [np.zeros((1, 1000), dtype=np.float32)]
    mock_model.post_proc.return_value = classifications if classifications is not None else []
    return mock_model


def _invoke(
    args: list[str],
    tmp_path: Path,
    mock_mgr: MagicMock,
    mock_backend: MagicMock | None = None,
) -> Result:
    from moment_to_action._cli import cli

    be = mock_backend or MagicMock()
    with patch("moment_to_action._cli.init_logging"):
        with patch("moment_to_action._cli.PathManager", return_value=_patched_pm(tmp_path)):
            with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                with patch(
                    "moment_to_action._cli.commands.cmd_model.cmd_run.ModelManager",
                    return_value=mock_mgr,
                ):
                    with patch(
                        "moment_to_action._cli.commands.cmd_model.cmd_run.ComputeBackend",
                        return_value=be,
                    ):
                        return CliRunner().invoke(cli, ["model", "run", *args])


def _write_image(path: Path) -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


@pytest.mark.unit
class TestModelRunCommand:
    """Tests for m2a model run."""

    def test_json_output_valid(self, tmp_path: Path) -> None:
        """--format json produces valid JSON to stdout."""
        img_path = tmp_path / "img.jpg"
        _write_image(img_path)
        det = _make_detection()
        mock_model = _make_mock_model([det])
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model

        result = _invoke(["yolo_v8", str(img_path)], tmp_path, mgr)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert data[0]["label"] == "person"
        assert data[0]["confidence"] == pytest.approx(0.9)

    def test_json_bbox_fields_present(self, tmp_path: Path) -> None:
        """JSON output includes bbox fields."""
        img_path = tmp_path / "img.jpg"
        _write_image(img_path)
        det = _make_detection()
        mock_model = _make_mock_model([det])
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model

        result = _invoke(["yolo_v8", str(img_path)], tmp_path, mgr)
        data = json.loads(result.output)
        assert "bbox" in data[0]

    def test_image_format_writes_file(self, tmp_path: Path) -> None:
        """--format image writes an output file."""
        img_path = tmp_path / "img.jpg"
        _write_image(img_path)
        mock_model = _make_mock_model()
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model

        out_path = tmp_path / "out.jpg"
        result = _invoke(
            ["yolo_v8", str(img_path), "--format", "image", "--output", str(out_path)],
            tmp_path,
            mgr,
        )
        assert result.exit_code == 0
        assert out_path.exists()

    def test_image_format_default_output_path(self, tmp_path: Path) -> None:
        """Default output path is <stem>_detections<ext> next to input."""
        img_path = tmp_path / "smoke.jpg"
        _write_image(img_path)
        mock_model = _make_mock_model()
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model

        result = _invoke(["yolo_v8", str(img_path), "--format", "image"], tmp_path, mgr)
        assert result.exit_code == 0
        expected = tmp_path / "smoke_detections.jpg"
        assert expected.exists()

    def test_bad_input_path_errors(self, tmp_path: Path) -> None:
        """Non-existent input path exits non-zero."""
        mgr = MagicMock()
        result = _invoke(["yolo_v8", str(tmp_path / "nonexistent.jpg")], tmp_path, mgr)
        assert result.exit_code != 0

    def test_unreadable_image_errors(self, tmp_path: Path) -> None:
        """File that cv2 cannot read as image exits non-zero."""
        bad_img = tmp_path / "bad.jpg"
        bad_img.write_bytes(b"not an image")
        mock_model = _make_mock_model()
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model

        result = _invoke(["yolo_v8", str(bad_img)], tmp_path, mgr)
        assert result.exit_code != 0

    def test_non_image_model_errors(self, tmp_path: Path) -> None:
        """Non-ImageModel subclass exits non-zero with a clear error."""
        img_path = tmp_path / "img.jpg"
        _write_image(img_path)
        # Plain MagicMock is not an ImageModel instance
        mock_model = MagicMock()
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model

        result = _invoke(["yolo_v8", str(img_path)], tmp_path, mgr)
        assert result.exit_code != 0
        assert "image model" in result.output.lower()

    def test_threshold_option_forwarded_to_get_model(self, tmp_path: Path) -> None:
        """--threshold is forwarded as confidence_threshold to get_model."""
        img_path = tmp_path / "img.jpg"
        _write_image(img_path)
        mock_model = _make_mock_model()
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model

        _invoke(["yolo_v8", str(img_path), "--threshold", "0.1"], tmp_path, mgr)

        mgr.get_model.assert_called_once()
        _, kwargs = mgr.get_model.call_args
        assert kwargs["confidence_threshold"] == pytest.approx(0.1)

    def test_backend_option_forwarded_to_compute_backend(self, tmp_path: Path) -> None:
        """--backend CPU passes ComputeUnit.CPU to ComputeBackend."""
        from moment_to_action.hardware import ComputeUnit

        img_path = tmp_path / "img.jpg"
        _write_image(img_path)
        mock_model = _make_mock_model()
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model

        mock_backend = MagicMock()
        with patch("moment_to_action._cli.init_logging"):
            with patch("moment_to_action._cli.PathManager", return_value=_patched_pm(tmp_path)):
                with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    with patch(
                        "moment_to_action._cli.commands.cmd_model.cmd_run.ModelManager",
                        return_value=mgr,
                    ):
                        with patch(
                            "moment_to_action._cli.commands.cmd_model.cmd_run.ComputeBackend",
                            return_value=mock_backend,
                        ) as cb_cls:
                            from moment_to_action._cli import cli

                            CliRunner().invoke(
                                cli, ["model", "run", "yolo_v8", str(img_path), "--backend", "CPU"]
                            )
                            cb_cls.assert_called_once_with(preferred_unit=ComputeUnit.CPU)

    def test_model_unloaded_even_on_error(self, tmp_path: Path) -> None:
        """model.unload() is called even when inference raises."""
        img_path = tmp_path / "img.jpg"
        _write_image(img_path)
        mock_model = _make_mock_model()
        mock_model.run.side_effect = RuntimeError("inference failed")
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model

        _invoke(["yolo_v8", str(img_path)], tmp_path, mgr)
        mock_model.unload.assert_called_once()

    def test_image_format_draws_detections(self, tmp_path: Path) -> None:
        """Image format draws detections onto the frame."""
        img_path = tmp_path / "img.jpg"
        _write_image(img_path)
        det = _make_detection(label="car", conf=0.85)
        mock_model = _make_mock_model([det])
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model

        out_path = tmp_path / "out.jpg"
        result = _invoke(
            ["yolo_v8", str(img_path), "--format", "image", "--output", str(out_path)],
            tmp_path,
            mgr,
        )
        assert result.exit_code == 0
        assert out_path.exists()
        out_img = cv2.imread(str(out_path))
        assert out_img is not None
        assert out_img.shape == (100, 100, 3)

    def test_classification_image_format_writes_file(self, tmp_path: Path) -> None:
        """--format image on a classification model writes annotated output."""
        img_path = tmp_path / "img.jpg"
        _write_image(img_path)
        cls = _make_classification()
        mock_model = _make_mock_classification_model([cls])
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model

        out_path = tmp_path / "out.jpg"
        result = _invoke(
            ["mobilenet_v2", str(img_path), "--format", "image", "--output", str(out_path)],
            tmp_path,
            mgr,
        )
        assert result.exit_code == 0
        assert out_path.exists()

    def test_classification_image_format_default_output_path(self, tmp_path: Path) -> None:
        """Default output path for classification adds _classifications suffix."""
        img_path = tmp_path / "photo.jpg"
        _write_image(img_path)
        mock_model = _make_mock_classification_model()
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model

        result = _invoke(["mobilenet_v2", str(img_path), "--format", "image"], tmp_path, mgr)
        assert result.exit_code == 0
        expected = tmp_path / "photo_classifications.jpg"
        assert expected.exists()


@pytest.mark.unit
class TestOverlayClassifications:
    """Tests for _overlay_classifications helper."""

    def test_returns_ndarray_with_same_shape(self) -> None:
        """_overlay_classifications returns an array with same shape as input frame."""
        from moment_to_action._cli.commands.cmd_model.cmd_run import _overlay_classifications

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cls = _make_classification()
        result = _overlay_classifications(frame, [cls])
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)

    def test_does_not_modify_original_frame(self) -> None:
        """_overlay_classifications returns a copy, leaving original unchanged."""
        from moment_to_action._cli.commands.cmd_model.cmd_run import _overlay_classifications

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cls = _make_classification()
        result = _overlay_classifications(frame, [cls])
        assert result is not frame

    def test_empty_classifications_returns_copy(self) -> None:
        """Empty classifications list returns copy of frame without modification."""
        from moment_to_action._cli.commands.cmd_model.cmd_run import _overlay_classifications

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = _overlay_classifications(frame, [])
        assert result is not frame
        np.testing.assert_array_equal(result, frame)
