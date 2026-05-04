"""Tests for YOLOv12 model variant resolution."""

from __future__ import annotations

import pytest

from moment_to_action.hardware import ComputeUnit
from moment_to_action.models._variants.yolov12 import get_yolov12_model_for_unit


@pytest.mark.unit
class TestYolov12VariantResolution:
    """Tests for per-unit YOLOv12 model selection."""

    def test_gpu_resolves_to_qnn_so(self) -> None:
        """GPU resolution returns the compiled QNN model library path."""
        model_path = get_yolov12_model_for_unit(unit=ComputeUnit.GPU)
        assert model_path.name == "libyolov8n_qnn.so"
        assert model_path.exists()

    def test_cpu_resolves_to_float32_tflite(self) -> None:
        """CPU resolution returns the float32 TFLite model path."""
        model_path = get_yolov12_model_for_unit(unit=ComputeUnit.CPU)
        assert model_path.name == "yolov8n_float32.tflite"
        assert model_path.exists()

    def test_npu_resolves_to_int8_tflite(self) -> None:
        """NPU resolution returns the int8 TFLite model path."""
        model_path = get_yolov12_model_for_unit(unit=ComputeUnit.NPU)
        assert model_path.name == "yolov8n_w8a8.tflite"
        assert model_path.exists()
