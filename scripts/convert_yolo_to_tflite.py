#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ultralytics",
#   "onnx>=1.12.0,<2.0.0",
#   "onnxslim>=0.1.71",
#   "onnx2tf>=1.26.3,<1.29.0",
#   "onnx_graphsurgeon>=0.3.26",
#   "sng4onnx>=1.0.1",
#   "tf_keras<=2.19.0",
# ]
# ///

"""Convert YOLO26n to TFLite for QCS6490 acceleration via ultralytics.

The converted ``model.tflite`` (float32) and ``model_int8.tflite`` are written
under ``src/moment_to_action/models/_vendored/yolo/``. ``YOLOStage`` and
``YOLOBenchmark`` use the INT8 variant on NPU and float32 on GPU.

Pass ``--imgsz 320`` to produce a 320x320 INT8 model (``model_int8_320.tflite``)
that fits within Hexagon HTP TCM constraints on the QCS6490.

Usage:

    uv run scripts/convert_yolo_to_tflite.py
    uv run scripts/convert_yolo_to_tflite.py --imgsz 320
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_YOLO_DIR = _REPO_ROOT / "src" / "moment_to_action" / "models" / "_vendored" / "yolo"

_MODEL_NAME = "yolo26n.pt"

_DEFAULT_IMGSZ = 640
_NPU_IMGSZ = 320


def _pick_tflite(candidates: list[Path], preferred_keyword: str) -> Path:
    if not candidates:
        msg = "ultralytics did not produce a .tflite file — check the output for errors."
        print(msg, file=sys.stderr)
        sys.exit(1)
    return next((f for f in candidates if preferred_keyword in f.name.lower()), candidates[0])


def convert(imgsz: int = 640) -> None:
    """Export yolo26n.pt → float32/int8 TFLite variants using ultralytics.

    Args:
        imgsz: Input image size (square).  Use 320 to produce a model that fits
               within Hexagon HTP TCM constraints on the QCS6490 — the 640x640
               model requires ~2.56 MB per tensor, exceeding the ~2 MB default
               VTCM allocation.
    """
    from ultralytics import YOLO

    suffix = f"_{imgsz}" if imgsz != _DEFAULT_IMGSZ else ""
    tflite_path = _YOLO_DIR / f"model{suffix}.tflite"
    tflite_int8_path = _YOLO_DIR / f"model_int8{suffix}.tflite"

    if tflite_path.exists() and tflite_int8_path.exists():
        print(
            f"TFLite models already exist at {tflite_path} and {tflite_int8_path} —"
            " skipping conversion."
        )
        print("Delete them first if you want to re-convert.")
        return

    print(f"Exporting {_MODEL_NAME} to float32/int8 TFLite (imgsz={imgsz}) …")

    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        os.chdir(tmp_path)
        try:
            model = YOLO(_MODEL_NAME)
            float32_export_path = model.export(format="tflite", imgsz=imgsz)
            int8_export_path = model.export(format="tflite", int8=True, imgsz=imgsz)
        finally:
            os.chdir(original_cwd)

        candidates = sorted(tmp_path.glob("**/*.tflite"))

        float32_src = Path(float32_export_path)
        if not float32_src.exists():
            float32_src = _pick_tflite(candidates, "float32")

        int8_src = Path(int8_export_path)
        if not int8_src.exists():
            int8_src = _pick_tflite(candidates, "int8")

        shutil.copy2(float32_src, tflite_path)
        shutil.copy2(int8_src, tflite_int8_path)

    print(f"Written:  {tflite_path}  ({tflite_path.stat().st_size // 1024} KB)")
    print(f"Written:  {tflite_int8_path}  ({tflite_int8_path.stat().st_size // 1024} KB)")
    if imgsz == _NPU_IMGSZ:
        print(
            "Note: 320x320 model targets Hexagon HTP NPU — fits within TCM constraints.\n"
            "Run `uv run python scripts/benchmark_model.py --model yolo --units npu` to verify."
        )
    else:
        print(
            "Run `uv run python scripts/benchmark_model.py --model yolo --units npu gpu` to verify."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        choices=[320, 640],
        help=(
            "Input image size (square). Use 320 for NPU (fits Hexagon TCM), "
            "640 for CPU/GPU (default)."
        ),
    )
    args = parser.parse_args()
    convert(imgsz=args.imgsz)
