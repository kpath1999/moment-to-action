"""Convert YOLO26n to TFLite for QCS6490 acceleration via ultralytics.

The converted ``model.tflite`` (float32) and ``model_int8.tflite`` are written
alongside the existing ``model.onnx`` in
``src/moment_to_action/models/_vendored/yolo/``. ``YOLOStage`` and
``YOLOBenchmark`` use the INT8 variant on NPU and float32 variant on GPU.

Requires ``ultralytics`` and its TFLite export dependencies (not in the default project
dependencies).  Pass them all explicitly so uv manages the install instead of letting
ultralytics attempt a ``pip install`` that fails on PEP 668 / externally-managed systems:

    uv run --with "ultralytics,onnx>=1.12.0,<2.0.0,onnxslim>=0.1.71,onnx2tf>=1.26.3,<1.29.0,onnx_graphsurgeon>=0.3.26,sng4onnx>=1.0.1,tf_keras<=2.19.0" python scripts/convert_yolo_to_tflite.py
"""

from __future__ import annotations

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
_TFLITE_PATH = _YOLO_DIR / "model.tflite"
_TFLITE_INT8_PATH = _YOLO_DIR / "model_int8.tflite"

_MODEL_NAME = "yolo26n.pt"


def _check_ultralytics() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print(
            "ultralytics is not installed.\n"
            'Run:  uv run --with "ultralytics,onnx>=1.12.0,<2.0.0,onnxslim>=0.1.71,'
            'onnx2tf>=1.26.3,<1.29.0,onnx_graphsurgeon>=0.3.26,sng4onnx>=1.0.1,'
            'tf_keras<=2.19.0" python scripts/convert_yolo_to_tflite.py',
            file=sys.stderr,
        )
        sys.exit(1)


def _pick_tflite(candidates: list[Path], preferred_keyword: str) -> Path:
    if not candidates:
        msg = "ultralytics did not produce a .tflite file — check the output for errors."
        print(msg, file=sys.stderr)
        sys.exit(1)
    return next((f for f in candidates if preferred_keyword in f.name.lower()), candidates[0])


def convert() -> None:
    """Export yolo26n.pt → float32/int8 TFLite variants using ultralytics."""
    _check_ultralytics()
    from ultralytics import YOLO

    if _TFLITE_PATH.exists() and _TFLITE_INT8_PATH.exists():
        print(
            f"TFLite models already exist at {_TFLITE_PATH} and {_TFLITE_INT8_PATH} —"
            " skipping conversion."
        )
        print("Delete them first if you want to re-convert.")
        return

    print(f"Exporting {_MODEL_NAME} to float32/int8 TFLite …")

    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        os.chdir(tmp_path)
        try:
            model = YOLO(_MODEL_NAME)
            float32_export_path = model.export(format="tflite")
            int8_export_path = model.export(format="tflite", int8=True)
        finally:
            os.chdir(original_cwd)

        candidates = sorted(tmp_path.glob("**/*.tflite"))

        float32_src = Path(float32_export_path)
        if not float32_src.exists():
            float32_src = _pick_tflite(candidates, "float32")

        int8_src = Path(int8_export_path)
        if not int8_src.exists():
            int8_src = _pick_tflite(candidates, "int8")

        shutil.copy2(float32_src, _TFLITE_PATH)
        shutil.copy2(int8_src, _TFLITE_INT8_PATH)

    print(f"Written:  {_TFLITE_PATH}  ({_TFLITE_PATH.stat().st_size // 1024} KB)")
    print(f"Written:  {_TFLITE_INT8_PATH}  ({_TFLITE_INT8_PATH.stat().st_size // 1024} KB)")
    print("Run `uv run python scripts/benchmark_model.py --model yolo --units npu gpu` to verify.")


if __name__ == "__main__":
    convert()
