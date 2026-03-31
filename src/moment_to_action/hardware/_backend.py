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
import time
from typing import TYPE_CHECKING

import attrs
import numpy as np

if TYPE_CHECKING:
    import os

    from moment_to_action.hardware._platforms._base import (
        InferenceBackend,
        ModelInput,
        PowerMonitor,
    )
    from moment_to_action.hardware._types import TorchExecutionPolicy

from moment_to_action.hardware._platforms._detection import Platform, detect_platform
from moment_to_action.hardware._platforms.qcs6490 import QCS6490Backend, QCS6490PowerMonitor
from moment_to_action.hardware._platforms.x86_64 import X86_64Backend, X86_64PowerMonitor
from moment_to_action.hardware._types import ComputeUnit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@attrs.frozen
class BenchmarkResult:
    """Latency statistics from a :meth:`ComputeBackend.benchmark` run.

    All times are in milliseconds.
    """

    mean_ms: float
    """Mean inference latency across all runs."""

    p50_ms: float
    """Median (50th percentile) latency."""

    p95_ms: float
    """95th percentile latency."""

    p99_ms: float
    """99th percentile latency."""

    min_ms: float
    """Minimum observed latency."""

    max_ms: float
    """Maximum observed latency."""

    compute_unit: str
    """Name of the compute unit used (e.g. ``"CPU"``, ``"NPU"``)."""

    n_runs: int
    """Number of inference runs performed."""


# ---------------------------------------------------------------------------
# Platform-aware factory helpers
# ---------------------------------------------------------------------------


def _make_power_monitor() -> PowerMonitor:
    """Return the power monitor appropriate for the detected platform."""
    match detect_platform():
        case Platform.QCS6490:
            return QCS6490PowerMonitor()
        case Platform.X86_64:
            return X86_64PowerMonitor()
        case Platform.MACOS_ARM64:
            # Reuse x86_64 monitor fallback heuristics for local CPU testing.
            return X86_64PowerMonitor()


def _make_backend(preferred_unit: ComputeUnit) -> InferenceBackend:
    """Return the platform backend for the detected platform."""
    match detect_platform():
        case Platform.QCS6490:
            return QCS6490Backend(preferred_unit=preferred_unit)
        case Platform.X86_64:
            return X86_64Backend()
        case Platform.MACOS_ARM64:
            # macOS arm64 currently uses the CPU-oriented runtime path.
            return X86_64Backend()


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
            "ComputeBackend: preferred=%s active=%s", preferred_unit.name, self.active_unit.name
        )

        # Log a warning if we didn't get our prefered backend
        if self.active_unit != self._preferred_unit:
            logger.warning(
                "Preferred compute backend unit %s not available, falling back to %s",
                preferred_unit.name,
                self.active_unit.name,
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

    def load_model(self, model_path: str | os.PathLike[str]) -> object:
        """Load a model (delegates to the platform backend).

        Args:
            model_path: Path to a ``.tflite`` or ``.onnx`` file.

        Returns:
            An opaque model handle — pass it back to :meth:`run`.
        """
        return self._backend.load_model(model_path)

    def run(self, model_handle: object, inputs: ModelInput) -> list[np.ndarray]:
        """Run inference (delegates to the platform backend).

        Args:
            model_handle: Handle returned by :meth:`load_model`.
            inputs: Single ndarray or name→tensor dict.

        Returns:
            List of output tensors.
        """
        return self._backend.run(model_handle, inputs)

    def get_input_details(self, model_handle: object) -> list[dict]:
        """Inspect model input slots (delegates to the platform backend).

        Args:
            model_handle: Handle returned by :meth:`load_model`.
        """
        return self._backend.get_input_details(model_handle)

    def get_output_details(self, model_handle: object) -> list[dict]:
        """Inspect model output slots (delegates to the platform backend).

        Args:
            model_handle: Handle returned by :meth:`load_model`.
        """
        return self._backend.get_output_details(model_handle)

    def resolve_torch_policy(self, requested: str = "auto") -> TorchExecutionPolicy:
        """Resolve torch device/dtype policy via the active platform backend.

        Args:
            requested: ``"auto"`` or a string accepted by ``torch.device``.

        Returns:
            A resolved torch execution policy.
        """
        return self._backend.resolve_torch_policy(requested)

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark(
        self,
        model_handle: object,
        inputs: ModelInput,
        n_runs: int = 20,
    ) -> BenchmarkResult:
        """Run inference *n_runs* times and return latency statistics.

        Args:
            model_handle: Handle returned by :meth:`load_model`.
            inputs: Inputs to pass on each run.
            n_runs: Number of inference repetitions.

        Returns:
            A :class:`BenchmarkResult` with latency percentiles and metadata.
        """
        latencies = np.empty(n_runs, dtype=np.float64)
        for i in range(n_runs):
            t = time.perf_counter()
            self._backend.run(model_handle, inputs)
            latencies[i] = (time.perf_counter() - t) * 1000.0

        return BenchmarkResult(
            mean_ms=float(np.mean(latencies)),
            p50_ms=float(np.percentile(latencies, 50)),
            p95_ms=float(np.percentile(latencies, 95)),
            p99_ms=float(np.percentile(latencies, 99)),
            min_ms=float(np.min(latencies)),
            max_ms=float(np.max(latencies)),
            compute_unit=self.active_unit.name,
            n_runs=n_runs,
        )
