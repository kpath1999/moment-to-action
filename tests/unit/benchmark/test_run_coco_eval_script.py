"""Unit tests for the COCO evaluation script entrypoint."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

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
    module = _load_script_module()
    parser = module._build_parser()
    args = parser.parse_args([])

    assert args.n_images == 500
    assert args.model == "all"
    assert args.edge_unit == "npu"
    assert args.conf_threshold == pytest.approx(0.25)
    assert "rf_detr_n" in parser._option_string_actions["--model"].choices
    assert "ssd_mobilenetv2" in parser._option_string_actions["--model"].choices
    assert "tinyclip_8m" in parser._option_string_actions["--model"].choices


@pytest.mark.unit
def test_main_all_models_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_script_module()

    class _FakeDataset:
        dataset_name = "coco_val2017"

        def __init__(self, n_images: int) -> None:
            self._n_images = n_images

        def images(self) -> list[Path]:
            return [Path(f"img_{idx}.jpg") for idx in range(self._n_images)]

        def captions(self, _name: str) -> list[str]:
            return ["a caption"]

    monkeypatch.setattr(module, "CocoDataset", _FakeDataset)
    monkeypatch.setattr(module, "ModelManager", object)
    monkeypatch.setattr(
        module,
        "_run_yolo_eval",
        lambda dataset, manager, unit, conf_threshold: {  # noqa: ARG005
            "map_50": 0.5,
            "map_75": 0.4,
            "inference_mean_ms": 11.0,
        },
    )
    monkeypatch.setattr(
        module,
        "_run_mobileclip_eval",
        lambda dataset, manager, unit: {  # noqa: ARG005
            "recall_at_1": 0.33,
            "inference_mean_ms": 22.0,
        },
    )
    monkeypatch.setattr(
        module,
        "_run_siglip_eval",
        lambda dataset, manager, unit: {  # noqa: ARG005
            "recall_at_1": 0.44,
            "inference_mean_ms": 33.0,
        },
    )

    monkeypatch.setattr(sys, "argv", ["run_coco_eval.py", "--n-images", "3", "--model", "all"])
    module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"] == "coco_val2017"
    assert payload["n_images"] == 3
    assert payload["yolo_v12_n"]["map_50"] == pytest.approx(0.5)
    assert payload["mobileclip_s2"]["recall_at_1"] == pytest.approx(0.33)
    assert payload["siglip"]["recall_at_1"] == pytest.approx(0.44)
    assert payload["rf_detr_n"]["status"] == "unsupported"
    assert payload["ssd_mobilenetv2"]["status"] == "unsupported"
    assert payload["tinyclip_8m"]["status"] == "unsupported"
