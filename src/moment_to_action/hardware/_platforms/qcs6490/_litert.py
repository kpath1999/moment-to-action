"""LiteRT (ai_edge_litert) backend for the QCS6490 platform.

This module provides:
- ``_load_interpreter`` — loads a TFLite model, raising on delegate failure
- ``_set_inputs`` — feeds tensors into an interpreter
- ``LiteRTBackend`` — NPU/GPU-accelerated inference backend

Design note — no silent CPU fallback:
    Backends in this layer intentionally raise on failure instead of
    retrying on CPU.  ``ComputeBackend`` (the orchestrator) catches those
    errors and decides whether to fall back.  Keeping the fallback logic
    in one place makes the system easier to reason about and test.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from moment_to_action.hardware._platforms._base import InferenceBackend, ModelInput
from moment_to_action.hardware._types import ComputeUnit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_interpreter(model_path: str, delegates: list) -> Any:
    """Load a TFLite model via ai_edge_litert (or tf.lite as fallback).

    Raises ``RuntimeError`` if delegate application fails instead of silently
    retrying on CPU — the caller decides the retry strategy.

    Args:
        model_path: Filesystem path to the ``.tflite`` model.
        delegates: List of loaded delegate objects (may be empty for CPU).

    Returns:
        An allocated interpreter handle.

    Raises:
        RuntimeError: If the delegate fails to apply to the model.
        ImportError: If neither ai_edge_litert nor tensorflow is installed.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter

        interp = Interpreter(model_path=model_path, experimental_delegates=delegates)
        logger.info("Loaded %s via ai_edge_litert", model_path)
    except RuntimeError as e:
        if delegates:
            # Raise so ComputeBackend can decide whether to retry on CPU.
            msg = f"Delegate failed: {e}"
            raise RuntimeError(msg) from e
        raise
    except ImportError:
        logger.warning("ai_edge_litert not installed, falling back to tf.lite")
        import tensorflow as tf

        # tf.lite is the legacy path; we don't hide delegate failures here either.
        interp = tf.lite.Interpreter(model_path=model_path, experimental_delegates=delegates)

    interp.allocate_tensors()
    return interp


def _set_inputs(interp: Any, inputs: ModelInput) -> None:
    """Feed input tensors into an interpreter.

    Single-input models (plain ndarray):
        The tensor is fed to input slot 0.

    Multi-input models (dict of name → tensor):
        Each tensor is matched by name to its input slot.  Dtype mismatches
        are caught early to produce actionable error messages.

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
                f"Input '{name}' dtype mismatch: got {tensor.dtype}, model expects {expected_dtype}"
            )
            raise TypeError(msg)
        interp.set_tensor(detail["index"], tensor)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class LiteRTBackend(InferenceBackend):
    """ai_edge_litert runtime — primary backend for NPU/GPU inference.

    Routes inference to the Hexagon HTP (NPU) or Adreno GPU via the QNN
    TFLite delegate.  Falls back to ``tf.lite`` when ``ai_edge_litert`` is
    not installed.

    Design note:
        ``_get_delegates`` raises on failure so that ``ComputeBackend`` can
        catch the error and fall back to CPU at the orchestration layer.
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

        interp = _load_interpreter(path, self._get_delegates())
        self._interpreter_cache[path] = interp
        logger.info("Loaded %s on %s", path, self._unit.name)
        return interp

    def _get_delegates(self) -> list:
        """Build the delegate list for the configured compute unit.

        Raises:
            RuntimeError: If the QNN delegate cannot be loaded.  The caller
                (``ComputeBackend._select_backend``) should catch this and
                fall back to CPU.
        """
        delegates = []
        try:
            from ai_edge_litert.interpreter import load_delegate

            if self._unit == ComputeUnit.NPU:
                qnn = load_delegate("/usr/lib/libQnnTFLiteDelegate.so")
                delegates.append(qnn)
                logger.info("QNN delegate loaded → Hexagon HTP/NPU")
        except Exception as e:
            # Log first so the caller's warning includes the original cause.
            msg = f"NPU delegate unavailable: {e}"
            raise RuntimeError(msg) from e
        return delegates

    def run(self, handle: Any, inputs: ModelInput) -> list[np.ndarray]:
        """Run inference and return all output tensors.

        Args:
            handle: Interpreter returned by :meth:`load_model`.
            inputs: Single ndarray or name→tensor dict.

        Returns:
            List of output tensors, one per output slot.
        """
        _set_inputs(handle, inputs)
        handle.invoke()
        return [handle.get_tensor(d["index"]) for d in handle.get_output_details()]

    def get_supported_unit(self) -> ComputeUnit:
        """Return the compute unit this backend targets."""
        return self._unit
