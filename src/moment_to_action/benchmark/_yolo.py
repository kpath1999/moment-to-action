from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from moment_to_action.benchmark._accuracy import compute_map50, parse_yolo_outputs
from moment_to_action.benchmark._base import ModelBenchmark
from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models import ModelID

if TYPE_CHECKING:
    from pathlib import Path

    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager

logger = logging.getLogger(__name__)


class YOLOBenchmark(ModelBenchmark):
    """Benchmark implementation for YOLOv8.

    Loads the TFLite variant (``YOLO_V8_TFLITE``) on accelerated compute units
    so inference runs through the LiteRT/QNN path instead of ONNX/CPU.  Falls
    back to the ONNX variant when the TFLite model has not yet been converted or
    when the active unit is CPU.

    Args:
        eval_image_paths: Optional list of image paths used for accuracy
            evaluation.  Each image is run through both the CPU/ONNX oracle and
            the current variant; mAP@50 against the oracle is returned.
    """

    def __init__(self, eval_image_paths: list[Path] | None = None) -> None:
        # Tracks the input tensor shape so _make_dummy_input can adapt after
        # _load_model detects whether the model wants NCHW or NHWC layout.
        self._input_shape: tuple[int, ...] = (1, 3, 640, 640)
        self._eval_image_paths: list[Path] = eval_image_paths or []

    @property
    def model_id(self) -> ModelID:
        return ModelID.YOLO_V8

    def _load_model(self, backend: ComputeBackend, manager: ModelManager) -> object:
        if backend.active_unit == ComputeUnit.NPU:
            # Prefer the 320x320 INT8 model on NPU: it is the only variant whose
            # largest intermediate tensor (~0.64 MB) fits within the Hexagon HTP
            # VTCM/TCM budget on QCS6490.  The 640x640 INT8 model requires
            # ~2.56 MB per tensor, which exceeds the default allocation and causes
            # a graph-prepare failure (tcm_migration error 17).
            for npu_model_id in (ModelID.YOLO_V8_TFLITE_INT8_320, ModelID.YOLO_V8_TFLITE_INT8):
                if manager.is_available(npu_model_id):
                    handle = backend.load_model(manager.get_path(npu_model_id))
                    details = backend.get_input_details(handle)
                    self._input_shape = tuple(int(d) for d in details[0]["shape"])
                    return handle

        # Prefer TFLite on accelerated units so inference routes through the
        # LiteRT/QNN delegate instead of onnxruntime CPU.
        if backend.active_unit != ComputeUnit.CPU and manager.is_available(ModelID.YOLO_V8_TFLITE):
            handle = backend.load_model(manager.get_path(ModelID.YOLO_V8_TFLITE))
            details = backend.get_input_details(handle)
            self._input_shape = tuple(int(d) for d in details[0]["shape"])
            return handle

        self._input_shape = (1, 3, 640, 640)
        return backend.load_model(manager.get_path(ModelID.YOLO_V8))

    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object:
        del handle
        shape = (batch_size, *self._input_shape[1:])
        return np.zeros(shape, dtype=np.float32)

    def _run_inference(self, handle: object, inputs: object, backend: ComputeBackend) -> None:
        if not isinstance(inputs, np.ndarray):
            msg = "YOLO benchmark expects ndarray inputs"
            raise TypeError(msg)
        backend.run(handle, inputs)

    def _evaluate_accuracy(
        self,
        handle: object,
        backend: ComputeBackend,
        manager: ModelManager,
    ) -> float | None:
        """Return mAP@50 vs CPU/ONNX oracle on the provided eval images.

        The CPU float32 ONNX model acts as the oracle (higher-fidelity
        reference).  For each eval image the oracle generates ground-truth
        boxes; the current variant's detections are then compared to those
        boxes via IoU-based matching.  If no eval images are configured the
        method returns ``None``.
        """
        if not self._eval_image_paths:
            return None

        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("opencv-python not installed — skipping YOLO accuracy evaluation")
            return None

        # Build a CPU backend oracle to generate ground-truth boxes.
        from moment_to_action.hardware import ComputeBackend

        cpu_backend = ComputeBackend(preferred_unit=ComputeUnit.CPU)
        oracle_handle = cpu_backend.load_model(manager.get_path(ModelID.YOLO_V8))

        # Detect the layout expected by the variant under evaluation.
        # For NHWC: shape is [1, H, W, C]; for NCHW: shape is [1, C, H, W].
        _n_dims = 4
        _channel = 3
        details = backend.get_input_details(handle)
        shape = details[0]["shape"]
        nhwc = len(shape) == _n_dims and int(shape[-1]) == _channel and int(shape[1]) != _channel
        if nhwc:
            h_in, w_in = int(shape[1]), int(shape[2])
        else:
            h_in, w_in = int(shape[2]), int(shape[3])

        oracle_preds: list[list[np.ndarray]] = []
        eval_preds: list[list[np.ndarray]] = []

        for img_path in self._eval_image_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning("Could not load eval image %s — skipping", img_path)
                continue

            # Preprocess for oracle (always NCHW 640x640 float32)
            _oracle_size = 640
            oracle_tensor = _preprocess_nchw(img_bgr, _oracle_size, _oracle_size)
            oracle_outputs = cpu_backend.run(oracle_handle, oracle_tensor)
            oracle_preds.append(parse_yolo_outputs(oracle_outputs))

            # Preprocess for the evaluated variant
            if nhwc:
                eval_tensor = _preprocess_nhwc(img_bgr, h_in, w_in)
            else:
                eval_tensor = _preprocess_nchw(img_bgr, h_in, w_in)
            eval_outputs = backend.run(handle, eval_tensor)
            # All boxes compared in oracle-space; scale eval boxes if model uses smaller size
            scale = float(_oracle_size) / h_in if h_in != _oracle_size else 1.0
            raw = parse_yolo_outputs(eval_outputs)
            if scale != 1.0:
                raw = [
                    np.array(
                        [b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale],
                        dtype=np.float32,
                    )
                    for b in raw
                ]
            eval_preds.append(raw)

        if not oracle_preds:
            return None

        return compute_map50(eval_preds, oracle_preds)


# ---------------------------------------------------------------------------
# Image preprocessing helpers — no Stage dependencies
# ---------------------------------------------------------------------------


def _letterbox_resize(
    img_bgr: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Resize *img_bgr* to *target_h* x *target_w* with letterboxing (aspect ratio preserved)."""
    src_h, src_w = img_bgr.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    import cv2  # type: ignore[import-untyped]

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas


def _preprocess_nchw(img_bgr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Return float32 RGB NCHW tensor ``[1, 3, h, w]`` normalised to [0, 1]."""
    canvas = _letterbox_resize(img_bgr, h, w)
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, normalise
    return np.expand_dims(rgb.transpose(2, 0, 1), 0)  # HWC→CHW→NCHW


def _preprocess_nhwc(img_bgr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Return float32 RGB NHWC tensor ``[1, h, w, 3]`` normalised to [0, 1]."""
    canvas = _letterbox_resize(img_bgr, h, w)
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return np.expand_dims(rgb, 0)
