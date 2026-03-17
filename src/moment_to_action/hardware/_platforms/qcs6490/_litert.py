"""LiteRT (ai_edge_litert) backend for the QCS6490 platform.

Provides :class:`LiteRTBackend` — the TFLite runtime for NPU/GPU/CPU inference.

Design note — no silent CPU fallback:
    This backend raises on delegate failure so that ``QCS6490Backend`` (the
    orchestrator) can decide whether to retry on CPU.  Keeping fallback logic
    in one place makes the system easier to reason about and test.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from moment_to_action.hardware._platforms._base import InferenceBackend, ModelInput
from moment_to_action.hardware._types import ComputeUnit

logger = logging.getLogger(__name__)

# Path to the Qualcomm QNN TFLite delegate shared library on-device.
_QNN_DELEGATE_PATH = "/usr/lib/libQnnTFLiteDelegate.so"

# Try to import ai_edge_litert at module load time.  On dev machines this
# package is absent, so we fall back to tf.lite (which ships with tensorflow).
try:
    from ai_edge_litert.interpreter import Interpreter as _Interpreter
    from ai_edge_litert.interpreter import load_delegate as _load_delegate

    _have_ai_edge_litert = True
except ImportError:
    # ai_edge_litert is absent (dev machine / older TF install) — fall back to
    # the legacy tf.lite runtime, which is assumed to always be present.
    from tensorflow.lite.python.interpreter import Interpreter as _Interpreter
    from tensorflow.lite.python.interpreter import load_delegate as _load_delegate

    _have_ai_edge_litert = False
    logger.warning("ai_edge_litert not available — using tf.lite as fallback")


class LiteRTBackend(InferenceBackend):
    """TFLite runtime that supports CPU, NPU, and GPU inference.

    Routes NPU/GPU inference to the Hexagon HTP or Adreno GPU via the QNN
    TFLite delegate.  When ``compute_unit=CPU`` is requested, no delegate is
    loaded and XNNPACK is used automatically.

    On machines where ``ai_edge_litert`` is absent, falls back to ``tf.lite``
    (the legacy runtime) transparently.
    """

    def __init__(self, compute_unit: ComputeUnit = ComputeUnit.NPU) -> None:
        self._unit = compute_unit
        self._interpreter_cache: dict[str, Any] = {}

    def load_model(self, path: str) -> Any:
        """Load a TFLite model, caching interpreters by path.

        Args:
            path: Filesystem path to the ``.tflite`` model.

        Returns:
            A cached or freshly allocated interpreter handle.

        Raises:
            RuntimeError: If the delegate fails to load or apply.
        """
        if path in self._interpreter_cache:
            logger.debug("Model cache hit: %s", path)
            return self._interpreter_cache[path]

        interp = self._load_interpreter(path, self._get_delegates())
        self._interpreter_cache[path] = interp
        logger.info("Loaded %s on %s", path, self._unit.name)
        return interp

    def run(self, handle: Any, inputs: ModelInput) -> list[np.ndarray]:
        """Run inference and return all output tensors.

        Args:
            handle: Interpreter returned by :meth:`load_model`.
            inputs: Single ndarray or name→tensor dict.

        Returns:
            List of output tensors, one per output slot.
        """
        self._set_inputs(handle, inputs)
        handle.invoke()
        return [handle.get_tensor(d["index"]) for d in handle.get_output_details()]

    def get_input_details(self, handle: Any) -> list[dict]:
        """Return the model's input tensor metadata.

        Args:
            handle: Interpreter returned by :meth:`load_model`.
        """
        return handle.get_input_details()

    def get_output_details(self, handle: Any) -> list[dict]:
        """Return the model's output tensor metadata.

        Args:
            handle: Interpreter returned by :meth:`load_model`.
        """
        return handle.get_output_details()

    def get_supported_unit(self) -> ComputeUnit:
        """Return the compute unit this backend targets."""
        return self._unit

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_delegates(self) -> list:
        """Build the delegate list for the configured compute unit.

        Returns an empty list for CPU — no delegate loading is attempted,
        and XNNPACK acceleration is applied automatically by the runtime.

        For NPU, loads the QNN TFLite delegate from ``_QNN_DELEGATE_PATH``.
        Raises ``RuntimeError`` on failure so ``QCS6490Backend`` can fall back.
        """
        # CPU path: no delegates needed.
        if self._unit == ComputeUnit.CPU:
            return []

        # NPU path: load QNN delegate.  Raise on failure so the caller can
        # decide whether to retry on CPU.
        if self._unit == ComputeUnit.NPU:
            try:
                qnn = _load_delegate(_QNN_DELEGATE_PATH)
            except Exception as e:
                msg = f"NPU delegate unavailable: {e}"
                raise RuntimeError(msg) from e
            else:
                logger.info("QNN delegate loaded → Hexagon HTP/NPU")
                return [qnn]

        # GPU and other units: no delegate implemented yet.
        return []

    def _load_interpreter(self, model_path: str, delegates: list) -> Any:
        """Load and allocate a TFLite interpreter.

        Uses ``ai_edge_litert`` when available, otherwise ``tf.lite`` — both
        are imported at module load time under the same ``_Interpreter`` name.
        Raises ``RuntimeError`` if a non-empty delegate list fails to apply,
        so the caller decides the retry strategy.

        Args:
            model_path: Filesystem path to the ``.tflite`` model.
            delegates: List of loaded delegate objects (may be empty for CPU).

        Returns:
            An allocated interpreter handle.

        Raises:
            RuntimeError: If a delegate fails to apply to the model.
        """
        try:
            interp = _Interpreter(model_path=model_path, experimental_delegates=delegates)
        except RuntimeError as e:
            if delegates:
                msg = f"Delegate failed: {e}"
                raise RuntimeError(msg) from e
            raise

        interp.allocate_tensors()
        return interp

    @staticmethod
    def _set_inputs(interp: Any, inputs: ModelInput) -> None:
        """Feed input tensors into an interpreter.

        For single-input models pass a plain ndarray (fed to slot 0).
        For multi-input models pass a name→tensor dict; each tensor is
        matched by name and dtype-checked before being set.

        Args:
            interp: An allocated LiteRT interpreter.
            inputs: Single ndarray or name→tensor mapping.

        Raises:
            KeyError: If a named input is not found in the model.
            TypeError: If a tensor dtype does not match the model's expected dtype.
        """
        input_details = interp.get_input_details()

        if isinstance(inputs, np.ndarray):
            interp.set_tensor(input_details[0]["index"], inputs)
            return

        name_to_detail = {d["name"]: d for d in input_details}
        for name, tensor in inputs.items():
            if name not in name_to_detail:
                available = list(name_to_detail.keys())
                msg = f"Input name '{name}' not found in model. Available: {available}"
                raise KeyError(msg)
            detail = name_to_detail[name]
            expected_dtype = detail["dtype"]
            if tensor.dtype != expected_dtype:
                msg = (
                    f"Input '{name}' dtype mismatch: "
                    f"got {tensor.dtype}, model expects {expected_dtype}"
                )
                raise TypeError(msg)
            interp.set_tensor(detail["index"], tensor)
