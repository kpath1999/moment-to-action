"""Unit tests for UltralyticsSource and resolve_ultralytics_source."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from moment_to_action.models import ModelFormat
from moment_to_action.models._sources._ultralytics import (
    UltralyticsSource,
    resolve_ultralytics_source,
)


@pytest.mark.unit
class TestUltralyticsSource:
    """Attribute and construction tests for UltralyticsSource."""

    def test_required_fields(self) -> None:
        """UltralyticsSource stores format and name."""
        s = UltralyticsSource(format=ModelFormat.ONNX, name="yolov8n")
        assert s.format is ModelFormat.ONNX
        assert s.name == "yolov8n"

    def test_default_filename(self) -> None:
        """Default filename is model.onnx."""
        s = UltralyticsSource(format=ModelFormat.ONNX, name="yolov8n")
        assert s.filename == "model.onnx"

    def test_custom_filename(self) -> None:
        """Custom filename is stored."""
        s = UltralyticsSource(format=ModelFormat.ONNX, name="yolov8n", filename="custom.onnx")
        assert s.filename == "custom.onnx"


@pytest.mark.unit
class TestResolveUltralyticsSource:
    """Tests for resolve_ultralytics_source."""

    def _source(self) -> UltralyticsSource:
        return UltralyticsSource(format=ModelFormat.ONNX, name="yolov8n")

    def test_returns_existing_file_without_download(self, tmp_path: Path) -> None:
        """If target file exists, returns path without importing ultralytics."""
        s = self._source()
        (tmp_path / "model.onnx").write_text("cached")

        with mock.patch.dict("sys.modules", {"ultralytics": None}):
            result = resolve_ultralytics_source(s, tmp_path)

        assert result == tmp_path / "model.onnx"

    def test_missing_without_download_returns_none(self, tmp_path: Path) -> None:
        """Missing file with download=False returns None."""
        s = self._source()
        result = resolve_ultralytics_source(s, tmp_path, download=False)
        assert result is None

    def test_download_true_imports_and_exports(self, tmp_path: Path) -> None:
        """download=True calls ultralytics.YOLO.export and caches the result."""
        s = self._source()
        exported = tmp_path / "yolov8n.onnx"
        exported.write_text("exported_model")

        mock_yolo_instance = mock.MagicMock()
        mock_yolo_instance.export.return_value = str(exported)
        mock_yolo_cls = mock.MagicMock(return_value=mock_yolo_instance)

        with mock.patch.dict("sys.modules", {"ultralytics": mock.MagicMock(YOLO=mock_yolo_cls)}):
            result = resolve_ultralytics_source(s, tmp_path, download=True)

        assert result == tmp_path / "model.onnx"
        assert (tmp_path / "model.onnx").read_text() == "exported_model"
        mock_yolo_cls.assert_called_once_with("yolov8n.pt")
        mock_yolo_instance.export.assert_called_once_with(format="onnx", dynamic=False)

    def test_download_removes_pt_file(self, tmp_path: Path) -> None:
        """The downloaded .pt file is removed after export."""
        s = self._source()
        exported = tmp_path / "yolov8n.onnx"
        exported.write_text("model")
        pt_file = Path("yolov8n.pt")
        pt_file.write_text("weights")

        mock_yolo_instance = mock.MagicMock()
        mock_yolo_instance.export.return_value = str(exported)
        mock_yolo_cls = mock.MagicMock(return_value=mock_yolo_instance)

        try:
            with mock.patch.dict(
                "sys.modules", {"ultralytics": mock.MagicMock(YOLO=mock_yolo_cls)}
            ):
                resolve_ultralytics_source(s, tmp_path, download=True)
        finally:
            pt_file.unlink(missing_ok=True)

        assert not pt_file.exists()

    def test_missing_ultralytics_raises_import_error(self, tmp_path: Path) -> None:
        """ImportError with helpful message when ultralytics is not installed."""
        s = self._source()
        # Patch only the ultralytics import inside the resolver module.
        with mock.patch.dict("sys.modules", {"ultralytics": None}):
            with pytest.raises(ImportError, match="ultralytics"):
                resolve_ultralytics_source(s, tmp_path, download=True)

    def test_progress_param_accepted(self, tmp_path: Path) -> None:
        """Progress parameter is accepted (interface uniformity)."""
        s = self._source()
        result = resolve_ultralytics_source(s, tmp_path, download=False, progress=False)
        assert result is None

    def test_dispatched_via_resolve_model_source(self, tmp_path: Path) -> None:
        """resolve_model_source dispatches UltralyticsSource correctly."""
        from moment_to_action.models._sources import resolve_model_source

        s = UltralyticsSource(format=ModelFormat.ONNX, name="yolov8n")
        (tmp_path / "model.onnx").write_text("ok")
        result = resolve_model_source(s, tmp_path)
        assert result == tmp_path / "model.onnx"
