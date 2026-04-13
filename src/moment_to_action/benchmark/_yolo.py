from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from moment_to_action.benchmark._base import ModelBenchmark
from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models import ModelID

if TYPE_CHECKING:
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager


class YOLOBenchmark(ModelBenchmark):
    """Benchmark implementation for YOLOv8.

    Loads the TFLite variant (``YOLO_V8_TFLITE``) on accelerated compute units
    so inference runs through the LiteRT/QNN path instead of ONNX/CPU.  Falls
    back to the ONNX variant when the TFLite model has not yet been converted or
    when the active unit is CPU.
    """

    def __init__(self) -> None:
        # Tracks the input tensor shape so _make_dummy_input can adapt after
        # _load_model detects whether the model wants NCHW or NHWC layout.
        self._input_shape: tuple[int, ...] = (1, 3, 640, 640)

    @property
    def model_id(self) -> ModelID:
        return ModelID.YOLO_V8

    def _load_model(self, backend: ComputeBackend, manager: ModelManager) -> object:
        if backend.active_unit == ComputeUnit.NPU and manager.is_available(
            ModelID.YOLO_V8_TFLITE_INT8
        ):
            handle = backend.load_model(manager.get_path(ModelID.YOLO_V8_TFLITE_INT8))
            details = backend.get_input_details(handle)
            self._input_shape = tuple(int(d) for d in details[0]["shape"])
            return handle

        # Prefer TFLite on accelerated units so inference routes through the
        # LiteRT/QNN delegate instead of onnxruntime CPU.
        if backend.active_unit != ComputeUnit.CPU and manager.is_available(
            ModelID.YOLO_V8_TFLITE
        ):
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
