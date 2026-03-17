"""Hardware Abstraction Layer — public entry point.

``ComputeBackend`` is the **only** class the rest of the codebase imports.
It is intentionally thin — three private fields and pure delegation:

    _preferred_unit   the unit the caller asked for
    _power_monitor    platform-appropriate power monitor
    _backend          platform-appropriate unified inference backend

All format routing (``.tflite`` vs ``.onnx``), sub-backend management,
and accelerator → CPU fallback logic live inside the platform backend
(e.g. :class:`QCS6490Backend`).  ``ComputeBackend`` just picks the right
platform at construction time and forwards every call.
"""

from __future__ import annotations

import logging
import platform
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from moment_to_action.hardware._platforms._base import (
        InferenceBackend,
        ModelInput,
        PowerMonitor,
    )

from moment_to_action.hardware._types import ComputeUnit

logger = logging.getLogger(__name__)

# CPU architectures where hardware accelerators (NPU/GPU) may be present.
# On other arches (x86_64 dev machines, CI) we tell the platform backend
# to target CPU directly — avoids noisy delegate-loading failures.
_ACCELERATOR_ARCHES: frozenset[str] = frozenset({"aarch64", "armv8l"})


# ---------------------------------------------------------------------------
# Platform-aware factory helpers
# ---------------------------------------------------------------------------


def _make_power_monitor() -> PowerMonitor:
    """Return the power monitor appropriate for the current host.

    Currently all platforms map to :class:`QCS6490PowerMonitor`, which
    gracefully falls back to static estimates when sysfs is absent.
    Extend with ``elif`` branches as new platforms are added.
    """
    from moment_to_action.hardware._platforms.qcs6490 import QCS6490PowerMonitor

    machine = platform.machine().lower()
    if machine not in _ACCELERATOR_ARCHES:
        logger.debug(
            "Platform %r: no accelerator expected — power monitor will use estimates",
            machine,
        )
    return QCS6490PowerMonitor()


def _make_backend(preferred_unit: ComputeUnit) -> InferenceBackend:
    """Return the platform backend for the current host.

    On non-accelerator architectures the backend is told to use CPU
    regardless of what *preferred_unit* says, so that delegate-loading
    failures never surface in dev/CI environments.
    """
    from moment_to_action.hardware._platforms.qcs6490 import QCS6490Backend

    machine = platform.machine().lower()

    # On dev machines force CPU to avoid noisy delegate warnings.
    if machine not in _ACCELERATOR_ARCHES and preferred_unit != ComputeUnit.CPU:
        logger.info("Platform %r: %s not available, using CPU", machine, preferred_unit.name)
        return QCS6490Backend(preferred_unit=ComputeUnit.CPU)

    return QCS6490Backend(preferred_unit=preferred_unit)


# ---------------------------------------------------------------------------
# ComputeBackend — public API
# ---------------------------------------------------------------------------


class ComputeBackend:
    """Hardware Abstraction Layer entry point.

    Models call this class — never LiteRT, ONNX, or SNPE directly.

    Usage::

        backend = ComputeBackend(preferred_unit=ComputeUnit.NPU)
        handle  = backend.load_model("mobileclip.tflite")
        outputs = backend.run(handle, {
            'serving_default_args_0:0': image_tensor,
            'serving_default_args_1:0': token_tensor,
        })

    Attributes:
        preferred_unit: The compute unit requested at construction time.
        power_monitor: Platform power monitor instance (read-only).
    """

    def __init__(self, preferred_unit: ComputeUnit = ComputeUnit.NPU) -> None:
        self._preferred_unit = preferred_unit
        self._power_monitor: PowerMonitor = _make_power_monitor()
        self._backend: InferenceBackend = _make_backend(preferred_unit)
        logger.info(
            "ComputeBackend: preferred=%s active=%s platform=%s",
            preferred_unit.name,
            self._backend.get_supported_unit().name,
            platform.machine(),
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def preferred_unit(self) -> ComputeUnit:
        """The compute unit originally requested."""
        return self._preferred_unit

    @property
    def power_monitor(self) -> PowerMonitor:
        """The platform power monitor."""
        return self._power_monitor

    @property
    def active_unit(self) -> ComputeUnit:
        """The compute unit actually in use (may differ from preferred after fallback)."""
        return self._backend.get_supported_unit()

    # ------------------------------------------------------------------
    # Delegation — every call forwards to the platform backend
    # ------------------------------------------------------------------

    def load_model(self, model_path: str) -> Any:
        """Load a model (delegates to the platform backend).

        Args:
            model_path: Path to a ``.tflite`` or ``.onnx`` file.

        Returns:
            An opaque model handle — pass it back to :meth:`run`.
        """
        return self._backend.load_model(model_path)

    def run(self, model_handle: Any, inputs: ModelInput) -> list[np.ndarray]:
        """Run inference (delegates to the platform backend).

        Args:
            model_handle: Handle returned by :meth:`load_model`.
            inputs: Single ndarray or name→tensor dict.

        Returns:
            List of output tensors.
        """
        return self._backend.run(model_handle, inputs)

    def get_input_details(self, model_handle: Any) -> list[dict]:
        """Inspect model input slots (TFLite-specific).

        Args:
            model_handle: Handle returned by :meth:`load_model`.
        """
        return model_handle.raw.get_input_details()

    def get_output_details(self, model_handle: Any) -> list[dict]:
        """Inspect model output slots (TFLite-specific).

        Args:
            model_handle: Handle returned by :meth:`load_model`.
        """
        return model_handle.raw.get_output_details()

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

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
        latencies: list[float] = []
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
