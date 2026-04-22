from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import attrs
import pytest

from moment_to_action.benchmark import CostProfile, VariantID, VariantProfile, VariantRegistry
from moment_to_action.hardware import ComputeUnit
from moment_to_action.models import ModelID


def _profile(
    model_id: ModelID,
    compute_unit: ComputeUnit,
    *,
    latency: float,
    accuracy: float | None,
    energy: float | None,
    accuracy_details: dict[str, float] | None = None,
) -> VariantProfile:
    return VariantProfile(
        variant_id=VariantID(model_id=model_id, compute_unit=compute_unit),
        accuracy=accuracy,
        load_latency_ms=1.0,
        inference_mean_ms=latency,
        inference_p50_ms=latency,
        inference_p95_ms=latency,
        inference_p99_ms=latency,
        peak_memory_mb=1.0,
        max_batch_size=1,
        hardware_target="x86_64",
        cost=CostProfile(power_mw=100.0, energy_per_inference_mj=energy),
        model_size_bytes=1,
        n_runs=1,
        profiled_at=datetime(2026, 1, 1, tzinfo=UTC),
        accuracy_details=accuracy_details,
    )


@pytest.mark.unit
def test_register_get_and_all_profiles() -> None:
    registry = VariantRegistry()
    profile = _profile(ModelID.YOLO_V8, ComputeUnit.CPU, latency=10.0, accuracy=0.5, energy=1.0)

    registry.register(profile)

    assert registry.get(profile.variant_id) == profile
    assert registry.all_profiles() == [profile]


@pytest.mark.unit
def test_query_filters() -> None:
    registry = VariantRegistry()
    p1 = _profile(ModelID.YOLO_V8, ComputeUnit.CPU, latency=10.0, accuracy=0.6, energy=1.0)
    p2 = _profile(ModelID.YOLO_V8, ComputeUnit.NPU, latency=5.0, accuracy=0.7, energy=0.6)
    p3 = _profile(ModelID.MOBILECLIP_S2, ComputeUnit.CPU, latency=12.0, accuracy=None, energy=None)

    registry.register(p1)
    registry.register(p2)
    registry.register(p3)

    assert registry.query(model_id=ModelID.YOLO_V8) == [p1, p2]
    assert registry.query(compute_unit=ComputeUnit.NPU) == [p2]
    assert registry.query(max_latency_ms=6.0) == [p2]
    assert registry.query(min_accuracy=0.65) == [p2]


@pytest.mark.unit
def test_best_variant_objectives() -> None:
    registry = VariantRegistry()
    p1 = _profile(ModelID.YOLO_V8, ComputeUnit.CPU, latency=10.0, accuracy=0.8, energy=2.0)
    p2 = _profile(ModelID.YOLO_V8, ComputeUnit.NPU, latency=4.0, accuracy=0.7, energy=0.5)
    registry.register(p1)
    registry.register(p2)

    assert registry.best_variant(ModelID.YOLO_V8, "latency") == p2
    assert registry.best_variant(ModelID.YOLO_V8, "accuracy") == p1
    assert registry.best_variant(ModelID.YOLO_V8, "efficiency") == p2

    with pytest.raises(ValueError, match="Unknown objective"):
        registry.best_variant(ModelID.YOLO_V8, "throughput")


@pytest.mark.unit
def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    registry = VariantRegistry(path=path)
    profile = _profile(
        ModelID.MOBILECLIP_S2, ComputeUnit.GPU, latency=3.0, accuracy=0.9, energy=0.2
    )
    registry.register(profile)

    registry.save()

    loaded = VariantRegistry(path=path)
    loaded.load()
    assert loaded.get(profile.variant_id) == profile


@pytest.mark.unit
def test_variant_registry_path_and_load_missing_noop(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    registry = VariantRegistry(path=path)
    assert registry.path == path

    registry.load()
    assert registry.all_profiles() == []


@pytest.mark.unit
def test_best_variant_returns_none_for_no_candidates_or_missing_metrics() -> None:
    registry = VariantRegistry()
    assert registry.best_variant(ModelID.YOLO_V8, "latency") is None

    p_no_acc = _profile(ModelID.YOLO_V8, ComputeUnit.CPU, latency=10.0, accuracy=None, energy=1.0)
    p_no_energy = _profile(
        ModelID.MOBILECLIP_S2,
        ComputeUnit.CPU,
        latency=10.0,
        accuracy=0.5,
        energy=None,
    )
    registry.register(p_no_acc)
    registry.register(p_no_energy)

    assert registry.best_variant(ModelID.YOLO_V8, "accuracy") is None
    assert registry.best_variant(ModelID.MOBILECLIP_S2, "efficiency") is None


@pytest.mark.unit
def test_query_filters_by_hardware_target() -> None:
    registry = VariantRegistry()
    cpu = _profile(ModelID.YOLO_V8, ComputeUnit.CPU, latency=10.0, accuracy=0.5, energy=1.0)
    npu = _profile(ModelID.YOLO_V8, ComputeUnit.NPU, latency=5.0, accuracy=0.6, energy=0.6)
    npu = attrs.evolve(npu, hardware_target="qcs6490")
    registry.register(cpu)
    registry.register(npu)

    matches = registry.query(model_id=ModelID.YOLO_V8, hardware_target="qcs6490")
    assert matches == [npu]
