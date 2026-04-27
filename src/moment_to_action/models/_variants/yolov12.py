"""Model variant resolution and conversion utilities for YOLOv12.

Provides functions to resolve and prepare the correct model file (FP32, FP16, QDQ INT8)
for a given compute unit, reusing logic from test_all_drivers.py.
"""

from __future__ import annotations

# Standard library
from pathlib import Path

# Third-party
import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)

# First-party
from moment_to_action.benchmark import CocoDataset
from moment_to_action.models import ModelID, ModelManager

N_CALIB = 128
YOLO_INPUT_HW = (640, 640)


def resolve_yolov12_paths(
    manager: ModelManager, fp32_override: str | None = None
) -> tuple[Path, Path, Path]:
    """Return (fp32_path, fp16_path, qdq_path) for YOLOv12 model."""
    fp32 = Path(fp32_override).resolve() if fp32_override else manager.get_path(ModelID.YOLO_V12_N)
    fp16 = fp32.with_name(f"{fp32.stem}_fp16.onnx")
    qdq = fp32.with_name(f"{fp32.stem}_qdq.onnx")
    return fp32, fp16, qdq


def ensure_fp16(fp32: Path, fp16: Path) -> Path:
    """Convert FP32 → FP16 if not already cached."""
    if fp16.exists():
        return fp16
    model_fp16 = float16.convert_float_to_float16(
        onnx.load(str(fp32)),
        keep_io_types=True,
    )
    onnx.save(model_fp16, str(fp16))
    return fp16


class _CocoCalibReader(CalibrationDataReader):
    def __init__(
        self,
        input_name: str,
        n_samples: int = N_CALIB,
        input_hw: tuple[int, int] = YOLO_INPUT_HW,
    ) -> None:
        import cv2

        self._input_name = input_name
        dataset = CocoDataset(n_images=n_samples)
        images = dataset.images()[:n_samples]
        self._samples = []
        h, w = input_hw
        for img_meta in images:
            if isinstance(img_meta, Path):
                img_path = img_meta
            elif isinstance(img_meta, str):
                img_path = Path(img_meta)
            elif isinstance(img_meta, dict):
                img_path = Path(img_meta.get("file_name") or img_meta.get("path", ""))
            else:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (w, h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = img.astype("float32") / 255.0
            tensor = tensor.transpose(2, 0, 1)
            tensor = tensor[None, ...]
            self._samples.append(tensor)
        if not self._samples:
            msg = "No loadable images for QDQ calibration."
            raise RuntimeError(msg)
        self._iter = iter(self._samples)

    def get_next(self) -> dict[str, object] | None:
        try:
            return {self._input_name: next(self._iter)}
        except StopIteration:
            return None


def ensure_qdq(fp32: Path, qdq: Path) -> Path:
    """Derive a QDQ INT8 model from the FP32 source if not already cached."""
    if qdq.exists():
        return qdq
    prep = fp32.with_name(fp32.stem + "_prep.onnx")
    quant_pre_process(
        input_model_path=str(fp32),
        output_model_path=str(prep),
        auto_merge=True,
        save_as_external_data=False,
    )
    input_name = onnx.load(str(prep)).graph.input[0].name
    reader = _CocoCalibReader(input_name=input_name, n_samples=N_CALIB)
    quantize_static(
        model_input=str(prep),
        model_output=str(qdq),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
    )
    prep.unlink(missing_ok=True)
    return qdq


def get_yolov12_model_for_unit(
    # manager: ModelManager,
    unit: str | None = None,
    fp32_override: str | None = None,  # noqa: ARG001
) -> Path:
    """Return the YOLOv8 ONNX model path for the given compute unit.

    Loads from models/_vendored/yolo/{unit}/yolov8_det-onnx-w8a8/yolov8_det.onnx.
    manager and fp32_override are unused, but kept for API compatibility.
    """
    from moment_to_action.hardware import ComputeUnit

    if unit is None:
        unit_str = "npu"
    elif (
        unit == ComputeUnit.CPU
        or str(unit).lower() == "cpu"
        or unit == ComputeUnit.GPU
        or str(unit).lower() == "gpu"
    ):
        unit_str = (
            str(unit).lower()
            if isinstance(unit, str)
            else ("cpu" if unit == ComputeUnit.CPU else "gpu")
        )
    elif unit == ComputeUnit.NPU or str(unit).lower() == "npu":
        unit_str = "npu"
    else:
        msg = f"Unknown compute unit: {unit}"
        raise ValueError(msg)

    base = Path(__file__).parent.parent / "_vendored" / "yolo"
    if unit_str in ("cpu", "gpu"):
        model_path = base / unit_str / "yolov8_det-onnx-w8a8" / "yolov8_det.onnx"
    elif unit_str == "npu":
        model_path = (
            base
            / unit_str
            / "yolov8_det-precompiled_qnn_onnx-w8a8-qualcomm_qcs6490"
            / "yolov8_det.onnx"
        )
    else:
        msg = f"Unsupported compute unit: {unit_str}"
        raise ValueError(msg)

    if not model_path.exists():
        err_msg = f"YOLOv8 model not found at {model_path}"
        raise FileNotFoundError(err_msg)
    return model_path.resolve()
