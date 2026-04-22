"""LiteRT (ai_edge_litert) backend for shared runtime inference.

Provides :class:`LiteRTBackend` — the platform-agnostic TFLite runtime for
CPU inference. Platform-specific subclasses (e.g., for QCS6490) override
:meth:`_get_delegates` to add NPU/GPU acceleration.

Design note — no silent CPU fallback:
    This backend raises on delegate failure so that the calling code (the
    orchestrator) can decide whether to retry on CPU.  Keeping fallback logic
    in one place makes the system easier to reason about and test.
"""

from __future__ import annotations

import logging
import os
from typing import cast

import numpy as np

from moment_to_action.hardware._platforms._base import InferenceBackend, ModelInput
from moment_to_action.hardware._types import ComputeUnit

logger = logging.getLogger(__name__)

# Try to import ai_edge_litert at module load time.  On dev machines this
# package is absent, so we fall back to tf.lite (which ships with tensorflow).
try:
    from ai_edge_litert.interpreter import Interpreter as _Interpreter
    from ai_edge_litert.interpreter import load_delegate as _load_delegate

    _have_ai_edge_litert = True
except ImportError:  # pragma: no cover
    # ai_edge_litert is absent (dev machine / older TF install) — fall back to
    # the legacy tf.lite runtime, which is assumed to always be present.
    from tensorflow.lite.python.interpreter import Interpreter as _Interpreter
    from tensorflow.lite.python.interpreter import load_delegate as _load_delegate  # noqa: F401

    _have_ai_edge_litert = False
    logger.warning("ai_edge_litert not available — using tf.lite as fallback")


class LiteRTBackend(InferenceBackend):
    """TFLite runtime that supports CPU inference and can be extended for accelerators.

    Base implementation provides CPU-only inference via XNNPACK.  Platform-
    specific subclasses override :meth:`_get_delegates` to add NPU/GPU support.

    On machines where ``ai_edge_litert`` is absent, falls back to ``tf.lite``
    (the legacy runtime) transparently.
    """

    def __init__(self, compute_unit: ComputeUnit = ComputeUnit.CPU) -> None:
        self._unit = compute_unit
        self._interpreter_cache: dict[str, object] = {}

    def load_model(self, path: str | os.PathLike[str]) -> object:
        """Load a TFLite model, caching interpreters by path.

        Args:
            path: Filesystem path to the ``.tflite`` model.

        Returns:
            A cached or freshly allocated interpreter handle.

        Raises:
            RuntimeError: If the delegate fails to load or apply.
        """
        path = os.fspath(path)  # normalise to str for cache key
        if path in self._interpreter_cache:
            logger.debug("Model cache hit: %s", path)
            return self._interpreter_cache[path]

        interp, actual_unit = self._load_interpreter(path)
        self._interpreter_cache[path] = interp
        logger.info("Loaded %s on %s", path, actual_unit.name)
        return interp

    def run(self, handle: object, inputs: ModelInput) -> list[np.ndarray]:
        """Run inference and return all output tensors.

        Args:
            handle: Interpreter returned by :meth:`load_model`.
            inputs: Single ndarray or name→tensor dict.

        Returns:
            List of output tensors, one per output slot.
        """
        interp = cast("_Interpreter", handle)
        self._set_inputs(interp, inputs)
        interp.invoke()
        return [interp.get_tensor(d["index"]) for d in interp.get_output_details()]

    def get_input_details(self, handle: object) -> list[dict]:
        """Return the model's input tensor metadata.

        Args:
            handle: Interpreter returned by :meth:`load_model`.
        """
        return cast("_Interpreter", handle).get_input_details()

    def get_output_details(self, handle: object) -> list[dict]:
        """Return the model's output tensor metadata.

        Args:
            handle: Interpreter returned by :meth:`load_model`.
        """
        return cast("_Interpreter", handle).get_output_details()

    def get_supported_unit(self) -> ComputeUnit:
        """Return the compute unit this backend is configured for."""
        return self._unit

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_delegates(self) -> list:
        """Return the delegate list for the configured compute unit.

        This base implementation returns no delegates (CPU-only).
        Platform-specific subclasses override this to add accelerator support.
        """
        return []

    def _load_interpreter(self, model_path: str) -> tuple[object, ComputeUnit]:
        """Load and allocate a TFLite interpreter.

        Calls :meth:`_get_delegates` internally to obtain the right delegate
        list for the configured compute unit.  Uses ``ai_edge_litert`` when
        available, otherwise ``tf.lite`` — both are imported at module load
        time under the same ``_Interpreter`` name.

        Args:
            model_path: Filesystem path to the ``.tflite`` model.

        Returns:
            A tuple containing the allocated interpreter handle and the
            compute unit effectively used by the runtime.

        Raises:
            RuntimeError: If a delegate fails to load or apply to the model.
        """
        logger.debug(
            "[load_interpreter] requesting delegates for %s on %s",
            model_path,
            self._unit.name,
        )
        delegates = self._get_delegates()
        logger.debug(
            "[load_interpreter] got %d delegate(s) for %s",
            len(delegates),
            model_path,
        )

        logger.debug("[load_interpreter] constructing Interpreter for %s", model_path)
        try:
            interp = _Interpreter(model_path=model_path, experimental_delegates=delegates)
        except RuntimeError as e:  # pragma: no cover
            if delegates:
                msg = f"Delegate failed during Interpreter construction: {e}"
                raise RuntimeError(msg) from e
            raise
        logger.debug("[load_interpreter] Interpreter constructed for %s", model_path)

        logger.debug("[load_interpreter] calling allocate_tensors() for %s", model_path)
        interp.allocate_tensors()
        logger.debug("[load_interpreter] allocate_tensors() complete for %s", model_path)

        # Log delegate partitioning stats if DEBUG level is enabled
        if logger.isEnabledFor(logging.DEBUG) and delegates:
            try:
                # Get execution plan - shows which nodes run on which delegate
                # Negative node IDs indicate delegate execution, non-negative = CPU
                exec_plan = interp._get_execution_plan()  # noqa: SLF001
                delegate_nodes = sum(1 for node_id in exec_plan if node_id < 0)
                cpu_nodes = len(exec_plan) - delegate_nodes
                total_nodes = len(exec_plan)

                if cpu_nodes > 0:
                    logger.warning(
                        "[load_interpreter] Heterogeneous execution for %s: "
                        "%d/%d nodes on delegate, %d on CPU fallback",
                        model_path,
                        delegate_nodes,
                        total_nodes,
                        cpu_nodes,
                    )
                else:
                    logger.debug(
                        "[load_interpreter] Full acceleration: %d/%d nodes on delegate",
                        delegate_nodes,
                        total_nodes,
                    )
            except (AttributeError, Exception):  # noqa: BLE001
                # _get_execution_plan may not exist in all TFLite versions
                logger.debug(
                    "[load_interpreter] _get_execution_plan unavailable, skipping delegate stats"
                )

        actual_unit = self._unit if delegates else ComputeUnit.CPU
        return interp, actual_unit

    @staticmethod
    def _set_inputs(interp: _Interpreter, inputs: ModelInput) -> None:
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
        # Fetch once — a list of dicts with 'index', 'name', 'dtype', 'shape', etc.
        input_details = interp.get_input_details()

        # Fast path: single-input model — skip name lookup and feed directly to slot 0.
        if isinstance(inputs, np.ndarray):
            interp.set_tensor(input_details[0]["index"], inputs)
            return

        # Multi-input path: build a name → detail map for O(1) lookup per tensor.
        name_to_detail = {d["name"]: d for d in input_details}
        for name, tensor in inputs.items():
            if name not in name_to_detail:
                available = list(name_to_detail.keys())
                msg = f"Input name '{name}' not found in model. Available: {available}"
                raise KeyError(msg)
            detail = name_to_detail[name]
            # Guard against silent precision loss — TFLite will happily accept the
            # wrong dtype and produce garbage outputs without raising on its own.
            expected_dtype = detail["dtype"]
            if tensor.dtype != expected_dtype:
                msg = (
                    f"Input '{name}' dtype mismatch: "
                    f"got {tensor.dtype}, model expects {expected_dtype}"
                )
                raise TypeError(msg)
            interp.set_tensor(detail["index"], tensor)
