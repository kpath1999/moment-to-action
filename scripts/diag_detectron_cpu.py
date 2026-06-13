#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = ["numpy", "opencv-python-headless", "moment-to-action"]
#
# [tool.uv.sources]
# moment-to-action = { path = "..", editable = true }
# ///
# ruff: noqa
"""On-device diagnostic: localize where Detectron2 CPU diverges from NPU.

Runs ONE image through the two-stage detector on NPU then CPU and prints
per-stage tensor stats + final detections, so we can see whether the CPU
collapse starts in the proposal generator (stage 1) or the ROI head
(stage 2) -- and whether the DLC artifact or the QAIRT CPU backend is at
fault.

NPU loads the .npu.bin context binaries (HTP); CPU loads the portable .dlc
(QAIRT CPU backend). To isolate artifact-vs-backend, also try forcing the
DLC onto HTP (rename/hide the .npu.bin, or run with FORCE_DLC=1 below).

Run on the QCS6490 device:  ./scripts/diag_detectron_cpu.py path/to/img.jpg
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from moment_to_action.config import load_config
from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.models import ModelID, ModelManager
from moment_to_action.paths import PathManager
from moment_to_action.qairt import QairtSDKManager

_VARIANT = "qcs6490_w8a16"


def _configure_qairt() -> None:
    path_manager = PathManager()
    config = load_config(path_manager.app_config_file)
    if config.qairt_sdk_path is None:
        print("QAIRT SDK path not configured — DLC backends unavailable.")
        return
    QairtSDKManager.from_app_config(config, path_manager).configure_env()


def _stats(name: str, arr: np.ndarray) -> str:
    a = np.asarray(arr, dtype=np.float64)
    return (
        f"{name}: shape={tuple(arr.shape)} dtype={arr.dtype} "
        f"min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f} std={a.std():.4f}"
    )


def _run_one(unit: ComputeUnit, img_path: Path, manager: ModelManager) -> None:
    import cv2

    print(
        f"\n===== backend={unit.name} (artifact: {'.npu.bin' if unit is ComputeUnit.NPU else '.dlc'}) ====="
    )
    model = manager.get_model(ModelID.DETECTRON2, variant=_VARIANT)
    backend = ComputeBackend(preferred_unit=unit)
    if backend.active_unit != unit:
        print(f"  SKIP: fell back to {backend.active_unit.name}")
        return
    model.load(backend)

    frame = cv2.imread(str(img_path))
    prepared = model.prepare(frame)
    print(_stats("input", prepared))

    out1 = backend.infer_dlc(model._handle_pg, prepared)
    for k in ("feature", "proposals", "score"):
        if k in out1:
            print(_stats(f"  pg.{k}", out1[k]))

    padded = model._filter_proposals(out1["proposals"], out1["score"])
    nz = int((padded[0].sum(axis=1) != 0).sum())
    print(f"  filtered proposals (non-zero rows): {nz}/{padded.shape[1]}")
    print(_stats("  padded_proposals", padded))

    feat = np.ascontiguousarray(np.transpose(out1["feature"], (0, 2, 3, 1)))
    out2 = backend.infer_dlc(model._handle_roi, {"features": feat, "proposals_boxes": padded})
    for k in ("boxes", "scores", "classes"):
        if k in out2:
            print(_stats(f"  roi.{k}", out2[k]))

    dets = model.post_proc([out2["boxes"], out2["scores"], out2["classes"]])
    print(f"  detections (>{model.confidence_threshold}): {len(dets)}")
    for d in dets[:10]:
        print(f"    {d.label:<14} {d.confidence:.3f}  {d.bbox}")

    model.unload()


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: diag_detectron_cpu.py <image>")
        raise SystemExit(2)
    _configure_qairt()
    img = Path(sys.argv[1])
    manager = ModelManager(PathManager())
    for unit in (ComputeUnit.NPU, ComputeUnit.CPU):
        try:
            _run_one(unit, img, manager)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR on {unit.name}: {exc!r}")


if __name__ == "__main__":
    main()
