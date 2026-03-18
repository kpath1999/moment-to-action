"""ONNX Runtime backend for the QCS6490 platform.

CPU-only by default.  ExecutionProvider can be swapped for GPU/NPU if an
appropriate ONNX EP is available, but that is not done here — ONNX models
are rare in this codebase and CPU is sufficient.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
import onnxruntime as ort

from moment_to_action.hardware._platforms._base import InferenceBackend, ModelInput
from moment_to_action.hardware._types import ComputeUnit

logger = logging.getLogger(__name__)


class ONNXBackend(InferenceBackend):
    """ONNX Runtime backend — used for YOLO and other ONNX models.

    Runs on CPU via ``CPUExecutionProvider``.  Sessions are cached by path
    to avoid redundant I/O on repeated ``load_model`` calls.
    """

    def __init__(self) -> None:
        self._session_cache: dict[str, object] = {}

    def load_model(self, path: str) -> object:
        """Load an ONNX model, caching sessions by path.

        Args:
            path: Filesystem path to the ``.onnx`` model file.

        Returns:
            A cached or freshly created ``onnxruntime.InferenceSession``.
        """
        if path in self._session_cache:
            logger.debug("ONNX cache hit: %s", path)
            return self._session_cache[path]

        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._session_cache[path] = session
        logger.info("Loaded %s via onnxruntime", path)
        return session

    def run(self, handle: object, inputs: ModelInput) -> list[np.ndarray]:
        """Run ONNX inference and return output tensors.

        Args:
            handle: Session returned by :meth:`load_model`.
            inputs: Single ndarray (mapped to the first input slot) or a
                name→tensor dict.

        Returns:
            List of output tensors.
        """
        session = cast("ort.InferenceSession", handle)
        input_details = session.get_inputs()
        feed = {input_details[0].name: inputs} if isinstance(inputs, np.ndarray) else inputs
        return session.run(None, feed)

    def get_input_details(self, handle: object) -> list[dict]:
        """Return the model's input metadata as a list of dicts.

        Args:
            handle: Session returned by :meth:`load_model`.
        """
        session = cast("ort.InferenceSession", handle)
        return [
            {"name": inp.name, "shape": inp.shape, "dtype": inp.type}
            for inp in session.get_inputs()
        ]

    def get_output_details(self, handle: object) -> list[dict]:
        """Return the model's output metadata as a list of dicts.

        Args:
            handle: Session returned by :meth:`load_model`.
        """
        session = cast("ort.InferenceSession", handle)
        return [
            {"name": out.name, "shape": out.shape, "dtype": out.type}
            for out in session.get_outputs()
        ]

    def get_supported_unit(self) -> ComputeUnit:
        """Return ``ComputeUnit.CPU``."""
        return ComputeUnit.CPU
