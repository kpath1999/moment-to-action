from __future__ import annotations

from unittest import mock

import pytest

from moment_to_action.benchmark import BenchmarkConfig, BenchmarkHarness, ModelBenchmark, VariantRegistry
from moment_to_action.hardware import ComputeUnit
from moment_to_action.models import ModelID


class _MockBenchmark(ModelBenchmark):
    def __init__(self, model_id: ModelID, profile: object) -> None:
        self._model_id = model_id
        self._profile = profile

    @property
    def model_id(self) -> ModelID:
        return self._model_id

    def profile(self, backend: object, manager: object, config: BenchmarkConfig | None = None) -> object:
        del backend, manager, config
        return self._profile

    def _load_model(self, backend: object, manager: object) -> object:
        raise NotImplementedError

    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object:
        raise NotImplementedError

    def _run_inference(self, handle: object, inputs: object, backend: object) -> None:
        raise NotImplementedError


@pytest.mark.unit
def test_run_all_registers_profiles() -> None:
    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.CPU
    manager = mock.MagicMock()
    registry = VariantRegistry()
    harness = BenchmarkHarness(backend=backend, manager=manager, registry=registry)

    profile1 = mock.MagicMock(variant_id=mock.MagicMock(model_id=ModelID.YOLO_V8))
    profile2 = mock.MagicMock(variant_id=mock.MagicMock(model_id=ModelID.MOBILECLIP_S2))

    harness.register_benchmark(_MockBenchmark(ModelID.YOLO_V8, profile1))
    harness.register_benchmark(_MockBenchmark(ModelID.MOBILECLIP_S2, profile2))

    results = harness.run_all()

    assert results == [profile1, profile2]


@pytest.mark.unit
def test_run_model_missing_registration_raises() -> None:
    harness = BenchmarkHarness(backend=mock.MagicMock(), manager=mock.MagicMock())

    with pytest.raises(RuntimeError, match="No benchmark registered"):
        harness.run_model(ModelID.QWEN3_4B)
