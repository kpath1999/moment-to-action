"""CPU-only LiteRT backend for the QCS6490 platform.

Used as the final fallback when NPU/GPU delegates are unavailable.
No hardware acceleration; relies on XNNPACK for CPU optimisation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from moment_to_action.hardware._platforms._base import InferenceBackend, ModelInput
from moment_to_action.hardware._platforms.qcs6490._litert import _load_interpreter, _set_inputs
from moment_to_action.hardware._types import ComputeUnit

logger = logging.getLogger(__name__)


class CPUBackend(InferenceBackend):
    """Pure-CPU TFLite backend.  No delegates, no hardware acceleration.

    Useful for development, CI environments, and as the final fallback when
    all accelerated backends fail.
    """

    def load_model(self, path: str) -> Any:
        """Load a TFLite model on CPU (no delegates).

        Args:
            path: Filesystem path to the ``.tflite`` model file.

        Returns:
            An allocated LiteRT interpreter handle.
        """
        return _load_interpreter(path, delegates=[])

    def run(self, handle: Any, inputs: ModelInput) -> list[np.ndarray]:
        """Run CPU inference and return output tensors.

        Args:
            handle: Interpreter returned by :meth:`load_model`.
            inputs: Single ndarray or name→tensor dict.

        Returns:
            List of output tensors.
        """
        _set_inputs(handle, inputs)
        handle.invoke()
        return [handle.get_tensor(d["index"]) for d in handle.get_output_details()]

    def get_supported_unit(self) -> ComputeUnit:
        """Return ``ComputeUnit.CPU``."""
        return ComputeUnit.CPU
