from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from moment_to_action.benchmark._base import ModelBenchmark
from moment_to_action.models import ModelID

if TYPE_CHECKING:
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager


class MobileCLIPBenchmark(ModelBenchmark):
    """Benchmark implementation for MobileCLIP-S2."""

    @property
    def model_id(self) -> ModelID:
        return ModelID.MOBILECLIP_S2

    def _load_model(self, backend: ComputeBackend, manager: ModelManager) -> object:
        return backend.load_model(manager.get_path(self.model_id))

    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object:
        del handle
        return {
            "serving_default_args_0:0": np.zeros((batch_size, 3, 256, 256), dtype=np.float32),
            "serving_default_args_1:0": np.zeros((batch_size, 77), dtype=np.int64),
        }

    def _run_inference(self, handle: object, inputs: object, backend: ComputeBackend) -> None:
        if not isinstance(inputs, dict):
            msg = "MobileCLIP benchmark expects dict inputs"
            raise TypeError(msg)
        backend.run(handle, inputs)
