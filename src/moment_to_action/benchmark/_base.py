from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import psutil

from moment_to_action.benchmark._types import BenchmarkConfig, CostProfile, VariantID, VariantProfile
from moment_to_action.hardware._platforms._detection import detect_platform

if TYPE_CHECKING:
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelID, ModelManager


class ModelBenchmark(ABC):
    """Abstract benchmark for a model family."""

    @property
    @abstractmethod
    def model_id(self) -> ModelID:
        """Model identifier benchmarked by this implementation."""

    def profile(
        self,
        backend: ComputeBackend,
        manager: ModelManager,
        config: BenchmarkConfig | None = None,
    ) -> VariantProfile:
        """Profile this model on the provided backend and return a variant profile."""
        cfg = config or BenchmarkConfig()

        t_load = time.perf_counter()
        handle = self._load_model(backend, manager)
        load_latency_ms = (time.perf_counter() - t_load) * 1000.0

        batch_size = cfg.batch_sizes[0] if cfg.batch_sizes else 1
        warmup_inputs = self._make_dummy_input(handle, batch_size=batch_size)
        for _ in range(cfg.n_warmup):
            self._run_inference(handle, warmup_inputs, backend)

        process = psutil.Process()
        peak_rss_mb = process.memory_info().rss / (1024.0 * 1024.0)
        latencies_ms = np.empty(cfg.n_runs, dtype=np.float64)

        run_inputs = self._make_dummy_input(handle, batch_size=batch_size)
        for idx in range(cfg.n_runs):
            t0 = time.perf_counter()
            self._run_inference(handle, run_inputs, backend)
            latencies_ms[idx] = (time.perf_counter() - t0) * 1000.0
            peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024.0 * 1024.0))

        accuracy = self._evaluate_accuracy(handle, backend, manager)
        max_batch_size = self._probe_max_batch_size(handle, backend)

        model_size_bytes = self._model_size_bytes(manager.get_path(self.model_id))
        mean_ms = float(np.mean(latencies_ms))

        power_mw: float | None = None
        try:
            power_mw = float(backend.resource_monitor.sample(backend.active_unit).power_mw)
        except (AttributeError, OSError, RuntimeError, ValueError):
            power_mw = None

        energy_mj = None if power_mw is None else (power_mw * mean_ms) / 1000.0

        return VariantProfile(
            variant_id=VariantID(model_id=self.model_id, compute_unit=backend.active_unit),
            accuracy=accuracy,
            load_latency_ms=load_latency_ms,
            inference_mean_ms=mean_ms,
            inference_p50_ms=float(np.percentile(latencies_ms, 50)),
            inference_p95_ms=float(np.percentile(latencies_ms, 95)),
            inference_p99_ms=float(np.percentile(latencies_ms, 99)),
            peak_memory_mb=peak_rss_mb,
            max_batch_size=max_batch_size,
            hardware_target=detect_platform().name.lower(),
            cost=CostProfile(
                power_mw=power_mw,
                energy_per_inference_mj=energy_mj,
            ),
            model_size_bytes=model_size_bytes,
            n_runs=cfg.n_runs,
            profiled_at=datetime.now(tz=UTC),
        )

    @abstractmethod
    def _load_model(self, backend: ComputeBackend, manager: ModelManager) -> object:
        """Load model and return runtime handle."""

    @abstractmethod
    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object:
        """Create dummy input(s) matching this model signature."""

    @abstractmethod
    def _run_inference(self, handle: object, inputs: object, backend: ComputeBackend) -> None:
        """Run one inference call for the given input payload."""

    def _evaluate_accuracy(
        self,
        handle: object,
        backend: ComputeBackend,
        manager: ModelManager,
    ) -> float | None:
        """Evaluate model accuracy for this variant if applicable."""
        del handle, backend, manager
        return None

    def _probe_max_batch_size(
        self,
        handle: object,
        backend: ComputeBackend,
        max_probe: int = 32,
    ) -> int:
        """Probe max supported batch size via incremental trial runs."""
        last_success = 0
        for batch_size in range(1, max_probe + 1):
            try:
                inputs = self._make_dummy_input(handle, batch_size=batch_size)
                self._run_inference(handle, inputs, backend)
            except Exception:  # noqa: BLE001
                break
            last_success = batch_size
        return max(last_success, 1)

    @staticmethod
    def _model_size_bytes(path: Path) -> int:
        """Calculate file or directory size in bytes."""
        if path.is_file():
            return path.stat().st_size
        return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())
