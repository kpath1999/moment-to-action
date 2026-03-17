"""Hardware Abstraction Layer — orchestrator.

``ComputeBackend`` is the single entry point that models call.  It handles:
- Platform detection and backend instantiation
- Fallback logic: NPU/GPU → CPU on failure
- ONNX model routing
- Benchmarking

Design rationale — fallback lives here, not in backends:
    Individual backends (``LiteRTBackend``, etc.) raise immediately when they
    cannot use their designated accelerator.  ``ComputeBackend`` catches those
    errors and decides whether to retry with a cheaper backend.  This keeps
    each backend simple and single-purpose while giving the orchestrator full
    visibility over the fallback chain.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from moment_to_action.hardware._platforms._base import InferenceBackend, ModelInput

from moment_to_action.hardware._platforms.qcs6490 import (
    CPUBackend,
    LiteRTBackend,
    ONNXBackend,
    QCS6490PowerMonitor,
)
from moment_to_action.hardware._types import ComputeUnit

logger = logging.getLogger(__name__)


def _make_power_monitor() -> QCS6490PowerMonitor:
    """Instantiate the power monitor appropriate for the current platform.

    Currently only QCS6490 is supported; more platforms can be added here
    as ``elif platform.machine() == ...`` branches.
    """
    # Future: add platform detection here for other chips.
    return QCS6490PowerMonitor()


class ComputeBackend:
    """Hardware Abstraction Layer entry point.

    Models call this class — never LiteRT, ONNX, or SNPE directly.  Swapping
    the underlying runtime or the chip requires changes only in this module
    and the platform-specific backends.

    Usage::

        backend = ComputeBackend(preferred_unit=ComputeUnit.NPU)
        handle  = backend.load_model("mobileclip.tflite")
        outputs = backend.run(handle, {
            'serving_default_args_0:0': image_tensor,
            'serving_default_args_1:0': token_tensor,
        })
        image_emb = outputs[1]   # [1, 512]
        text_emb  = outputs[0]   # [1, 512]

    Attributes:
        preferred_unit: The compute unit requested at construction time.
        power_monitor: Platform power monitor instance.
    """

    def __init__(self, preferred_unit: ComputeUnit = ComputeUnit.NPU) -> None:
        self.preferred_unit = preferred_unit
        self.power_monitor = _make_power_monitor()
        # _select_backend handles fallback internally; _backend is always valid.
        self._backend: InferenceBackend = self._select_backend(preferred_unit)
        self._onnx_backend = ONNXBackend()
        logger.info("ComputeBackend: active unit = %s", self._backend.get_supported_unit().name)

    def _select_backend(self, unit: ComputeUnit) -> InferenceBackend:
        """Try to create the requested backend; fall back to CPU on failure.

        Backends raise ``RuntimeError`` when their accelerator is unavailable
        (e.g. missing delegate library, unsupported model format).  We catch
        those errors here and gracefully degrade to CPU rather than crashing
        the whole inference pipeline.

        Args:
            unit: The desired compute unit.

        Returns:
            The best available ``InferenceBackend`` for *unit*.
        """
        try:
            if unit == ComputeUnit.NPU:
                return LiteRTBackend(compute_unit=ComputeUnit.NPU)
            if unit == ComputeUnit.GPU:
                return LiteRTBackend(compute_unit=ComputeUnit.GPU)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Backend for %s unavailable (%s) — falling back to CPU",
                unit.name,
                e,
            )
        return CPUBackend()

    def load_model(self, model_path: str) -> Any:
        """Load a model, routing ONNX files to the ONNX backend.

        If the primary backend (NPU/GPU) fails to load the model, the call is
        retried on CPU so that inference can continue.

        Args:
            model_path: Filesystem path to the model file (``.tflite`` or ``.onnx``).

        Returns:
            An opaque model handle suitable for :meth:`run`.
        """
        if model_path and model_path.endswith(".onnx"):
            return self._onnx_backend.load_model(model_path)

        try:
            return self._backend.load_model(model_path)
        except Exception as e:
            if not isinstance(self._backend, CPUBackend):
                logger.warning(
                    "Model load failed on %s (%s) — retrying on CPU",
                    self._backend.get_supported_unit().name,
                    e,
                )
                self._backend = CPUBackend()
                return self._backend.load_model(model_path)
            raise

    def run(self, model_handle: Any, inputs: ModelInput) -> list[np.ndarray]:
        """Run inference, routing to the correct backend.

        ONNX sessions are detected by type and routed to ``_onnx_backend``.

        Args:
            model_handle: Handle returned by :meth:`load_model`.
            inputs: Single ndarray or name→tensor dict.

        Returns:
            List of output tensors.
        """
        try:
            import onnxruntime as ort

            if isinstance(model_handle, ort.InferenceSession):
                return self._onnx_backend.run(model_handle, inputs)
        except ImportError:
            pass
        return self._backend.run(model_handle, inputs)

    def get_input_details(self, model_handle: Any) -> list[dict]:
        """Inspect model input slots.

        Args:
            model_handle: Handle returned by :meth:`load_model`.

        Returns:
            List of input detail dicts (index, name, shape, dtype).
        """
        return model_handle.get_input_details()

    def get_output_details(self, model_handle: Any) -> list[dict]:
        """Inspect model output slots.

        Args:
            model_handle: Handle returned by :meth:`load_model`.

        Returns:
            List of output detail dicts.
        """
        return model_handle.get_output_details()

    @property
    def active_unit(self) -> ComputeUnit:
        """The compute unit of the currently active backend."""
        return self._backend.get_supported_unit()

    def benchmark(
        self,
        model_handle: Any,
        inputs: ModelInput,
        n_runs: int = 20,
    ) -> dict:
        """Run inference *n_runs* times and return latency statistics.

        Args:
            model_handle: Handle returned by :meth:`load_model`.
            inputs: Inputs to pass on each run.
            n_runs: Number of inference repetitions.

        Returns:
            Dict with keys ``mean_ms``, ``p50_ms``, ``p95_ms``, ``p99_ms``,
            ``min_ms``, ``max_ms``, ``compute_unit``, ``n_runs``.
        """
        latencies = []
        for _ in range(n_runs):
            t = time.perf_counter()
            self._backend.run(model_handle, inputs)
            latencies.append((time.perf_counter() - t) * 1000)

        arr = np.array(latencies)
        return {
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "compute_unit": self.active_unit.name,
            "n_runs": n_runs,
        }
