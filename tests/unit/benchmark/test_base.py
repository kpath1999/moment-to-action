from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from moment_to_action.benchmark import BenchmarkConfig, ModelBenchmark
from moment_to_action.hardware import ComputeUnit
from moment_to_action.models import ModelID


class _DummyBenchmark(ModelBenchmark):
    @property
    def model_id(self) -> ModelID:
        return ModelID.YOLO_V12_N

    def _load_model(self, backend: object, manager: object) -> object:
        del backend, manager
        return object()

    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object:
        del handle
        return np.zeros((batch_size, 1), dtype=np.float32)

    def _run_inference(self, handle: object, inputs: object, backend: object) -> None:
        del handle, inputs, backend


@pytest.mark.unit
def test_profile_collects_metrics(tmp_path: Path) -> None:
    benchmark = _DummyBenchmark()

    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.CPU
    backend.resource_monitor.sample.return_value = mock.MagicMock(power_mw=900.0)

    manager = mock.MagicMock()
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"abc")
    manager.get_path.return_value = model_file

    process = mock.MagicMock()
    process.memory_info.side_effect = [
        mock.MagicMock(rss=100 * 1024 * 1024),
        mock.MagicMock(rss=110 * 1024 * 1024),
        mock.MagicMock(rss=120 * 1024 * 1024),
    ]

    with (
        mock.patch(
            "moment_to_action.benchmark._benchmarks._base.psutil.Process",
            return_value=process,
        ),
        mock.patch("moment_to_action.benchmark._benchmarks._base.detect_platform") as mock_platform,
    ):
        mock_platform.return_value = mock.MagicMock(name="platform")
        mock_platform.return_value.name = "X86_64"
        profile = benchmark.profile(
            backend=backend,
            manager=manager,
            config=BenchmarkConfig(n_warmup=1, n_runs=2, batch_sizes=[1]),
        )

    assert profile.variant_id.model_id == ModelID.YOLO_V12_N
    assert profile.n_runs == 2
    assert profile.model_size_bytes == 3
    assert profile.peak_memory_mb >= 100.0


@pytest.mark.unit
def test_probe_max_batch_size_stops_on_error() -> None:
    benchmark = _DummyBenchmark()

    backend = mock.MagicMock()

    call_count = {"n": 0}

    def raising_run(handle: object, inputs: object, backend_: object) -> None:
        del handle, inputs, backend_
        call_count["n"] += 1
        if call_count["n"] >= 3:
            msg = "oom"
            raise RuntimeError(msg)

    benchmark._run_inference = raising_run  # type: ignore[method-assign, assignment]
    batch = benchmark._probe_max_batch_size(object(), backend, max_probe=10)
    assert batch == 2


@pytest.mark.unit
def test_profile_tolerates_resource_monitor_sampling_failure(tmp_path: Path) -> None:
    benchmark = _DummyBenchmark()

    backend = mock.MagicMock()
    backend.active_unit = ComputeUnit.CPU
    backend.resource_monitor.sample.side_effect = RuntimeError("monitor down")

    manager = mock.MagicMock()
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"abc")
    manager.get_path.return_value = model_file

    process = mock.MagicMock()
    process.memory_info.side_effect = [
        mock.MagicMock(rss=100 * 1024 * 1024),
        mock.MagicMock(rss=105 * 1024 * 1024),
    ]

    with (
        mock.patch(
            "moment_to_action.benchmark._benchmarks._base.psutil.Process",
            return_value=process,
        ),
        mock.patch("moment_to_action.benchmark._benchmarks._base.detect_platform") as mock_platform,
    ):
        mock_platform.return_value = mock.MagicMock(name="platform")
        mock_platform.return_value.name = "X86_64"
        profile = benchmark.profile(
            backend=backend,
            manager=manager,
            config=BenchmarkConfig(n_warmup=0, n_runs=1, batch_sizes=[1]),
        )

    assert profile.cost.power_mw is None
    assert profile.cost.energy_per_inference_mj is None


@pytest.mark.unit
def test_model_size_bytes_for_directory(tmp_path: Path) -> None:
    benchmark = _DummyBenchmark()
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()
    (model_dir / "a.bin").write_bytes(b"abc")
    (model_dir / "b.bin").write_bytes(b"de")

    assert benchmark._model_size_bytes(model_dir) == 5
