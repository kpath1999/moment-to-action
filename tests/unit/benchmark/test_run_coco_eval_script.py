"""Unit tests for the COCO evaluation script entrypoint."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_script_module() -> ModuleType:
    script_path = Path(__file__).parents[3] / "scripts" / "run_coco_eval.py"
    spec = importlib.util.spec_from_file_location("run_coco_eval_test_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_parser_defaults() -> None:
    """Parser defaults should match the expected COCO benchmark baseline."""
    module = _load_script_module()
    parser = module._build_parser()
    args = parser.parse_args([])

    assert args.n_images == 500
    assert args.model == "both"
    assert args.oracle_unit == "gpu"
    assert args.edge_unit == "npu"
    assert args.conf_threshold == pytest.approx(0.25)
    assert args.output is None


@pytest.mark.unit
def test_main_skip_oracle_and_write_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Main should skip oracle generation and emit/write combined model results."""
    module = _load_script_module()

    class _FakeDataset:
        dataset_name = "coco_val2017"

        def __init__(self, n_images: int) -> None:
            self._n_images = n_images

        def images(self) -> list[Path]:
            return [Path(f"img_{idx}.jpg") for idx in range(self._n_images)]

    output_path = tmp_path / "coco-results.json"

    oracle_calls: list[tuple[object, object, object]] = []

    def _fake_model_manager() -> object:
        return object()

    def _fake_run_yolo_eval(
        dataset: object,
        manager: object,
        unit: object,
        conf_threshold: float,
    ) -> dict[str, float]:
        del dataset, manager, unit
        return {
            "accuracy": 0.33,
            "map_50_95": 0.33,
            "conf_threshold": conf_threshold,
        }

    def _fake_run_mobileclip_eval(
        dataset: object,
        manager: object,
        unit: object,
    ) -> dict[str, float]:
        del dataset, manager, unit
        return {
            "accuracy": 0.44,
            "recall_at_1": 0.44,
        }

    monkeypatch.setattr(module, "CocoDataset", _FakeDataset)
    monkeypatch.setattr(module, "ModelManager", _fake_model_manager)
    monkeypatch.setattr(
        module,
        "OracleStore",
        lambda dataset_name: SimpleNamespace(
            path=tmp_path / f"oracle_{dataset_name}.json",
            load=lambda: None,
        ),
    )
    monkeypatch.setattr(
        module,
        "_run_oracle_passes",
        lambda dataset, manager, unit: oracle_calls.append((dataset, manager, unit)),
    )
    monkeypatch.setattr(module, "_run_yolo_eval", _fake_run_yolo_eval)
    monkeypatch.setattr(module, "_run_mobileclip_eval", _fake_run_mobileclip_eval)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_coco_eval.py",
            "--n-images",
            "3",
            "--model",
            "both",
            "--skip-oracle",
            "--output",
            str(output_path),
        ],
    )

    module.main()
    assert oracle_calls == []

    stdout_payload = json.loads(capsys.readouterr().out)
    assert stdout_payload["dataset"] == "coco_val2017"
    assert stdout_payload["n_images"] == 3
    assert stdout_payload["yolo"]["accuracy"] == pytest.approx(0.33)
    assert stdout_payload["mobileclip"]["accuracy"] == pytest.approx(0.44)

    written_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert written_payload == stdout_payload
