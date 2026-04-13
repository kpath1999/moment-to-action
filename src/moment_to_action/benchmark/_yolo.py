from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from moment_to_action.benchmark._base import ModelBenchmark
from moment_to_action.models import ModelID

if TYPE_CHECKING:
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager


class YOLOBenchmark(ModelBenchmark):
    """Benchmark implementation for YOLOv8."""

    @property
    def model_id(self) -> ModelID:
        return ModelID.YOLO_V8

    def _load_model(self, backend: ComputeBackend, manager: ModelManager) -> object:
        return backend.load_model(manager.get_path(self.model_id))

    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object:
        del handle
        return np.zeros((batch_size, 3, 640, 640), dtype=np.float32)

    def _run_inference(self, handle: object, inputs: object, backend: ComputeBackend) -> None:
        if not isinstance(inputs, np.ndarray):
            msg = "YOLO benchmark expects ndarray inputs"
            raise TypeError(msg)
        backend.run(handle, inputs)
