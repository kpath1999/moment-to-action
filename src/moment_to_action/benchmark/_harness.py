from __future__ import annotations

from typing import TYPE_CHECKING

from moment_to_action.benchmark._variant_registry import VariantRegistry

if TYPE_CHECKING:
    from moment_to_action.benchmark._benchmarks._base import ModelBenchmark
    from moment_to_action.benchmark._types import BenchmarkConfig, VariantProfile
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelID, ModelManager


class BenchmarkHarness:
    """Orchestrates benchmark execution and registry persistence."""

    def __init__(
        self,
        backend: ComputeBackend,
        manager: ModelManager,
        registry: VariantRegistry | None = None,
    ) -> None:
        self._backend = backend
        self._manager = manager
        self._registry = registry or VariantRegistry()
        self._benchmarks: dict[ModelID, ModelBenchmark] = {}

    @property
    def registry(self) -> VariantRegistry:
        """Access the backing variant registry."""
        return self._registry

    def register_benchmark(self, benchmark: ModelBenchmark) -> None:
        """Register a model benchmark implementation."""
        self._benchmarks[benchmark.model_id] = benchmark

    def run_all(self, config: BenchmarkConfig | None = None) -> list[VariantProfile]:
        """Run all registered benchmarks and return collected profiles."""
        profiles: list[VariantProfile] = []
        for benchmark in self._benchmarks.values():
            profile = benchmark.profile(self._backend, self._manager, config=config)
            self._registry.register(profile)
            profiles.append(profile)
        return profiles

    def run_model(self, model_id: ModelID, config: BenchmarkConfig | None = None) -> VariantProfile:
        """Run a single model benchmark and return its profile."""
        benchmark = self._benchmarks.get(model_id)
        if benchmark is None:
            msg = f"No benchmark registered for model: {model_id.value}"
            raise RuntimeError(msg)

        profile = benchmark.profile(self._backend, self._manager, config=config)
        self._registry.register(profile)
        return profile
