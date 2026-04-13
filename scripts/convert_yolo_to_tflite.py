"""Convert the vendored YOLOv8 ONNX model to TFLite for QCS6490 acceleration.

The converted ``model.tflite`` is written alongside the existing ``model.onnx``
in ``src/moment_to_action/models/_vendored/yolo/``.  Once present, ``YOLOStage``
and ``YOLOBenchmark`` automatically route accelerated inference through the
LiteRT/QNN delegate instead of onnxruntime/CPU.

Requires ``onnx2tf`` (not in the default project dependencies):

    uv run --with onnx2tf python scripts/convert_yolo_to_tflite.py

``onnx2tf`` inserts an input-transposition node so the TFLite model accepts
NHWC tensors ``[1, 640, 640, 3]`` — matching TFLite's native layout — even
though the source ONNX uses NCHW.  The stage handles the transposition
automatically by inspecting the model's input_details at load time.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_YOLO_DIR = _REPO_ROOT / "src" / "moment_to_action" / "models" / "_vendored" / "yolo"
_ONNX_PATH = _YOLO_DIR / "model.onnx"
_TFLITE_PATH = _YOLO_DIR / "model.tflite"


def _check_onnx2tf() -> None:
    try:
        import onnx2tf  # noqa: F401
    except ImportError:
        print(
            "onnx2tf is not installed.\n"
            "Run:  uv run --with onnx2tf python scripts/convert_yolo_to_tflite.py",
            file=sys.stderr,
        )
        sys.exit(1)


def convert() -> None:
    """Convert model.onnx → model.tflite using onnx2tf."""
    _check_onnx2tf()
    import onnx2tf

    if not _ONNX_PATH.exists():
        print(f"Source model not found: {_ONNX_PATH}", file=sys.stderr)
        sys.exit(1)

    if _TFLITE_PATH.exists():
        print(f"TFLite model already exists at {_TFLITE_PATH} — skipping conversion.")
        print("Delete it first if you want to re-convert.")
        return

    print(f"Converting {_ONNX_PATH} → {_TFLITE_PATH} …")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        onnx2tf.convert(
            input_onnx_file_path=str(_ONNX_PATH),
            output_folder_path=str(tmp_path),
            # Keep float32 — quantisation can be done separately if required.
            output_integer_quantized_tflite=False,
            non_verbose=True,
        )
        # onnx2tf writes <model_stem>_float32.tflite into the output folder.
        candidates = sorted(tmp_path.glob("*.tflite"))
        if not candidates:
            print(
                "onnx2tf did not produce a .tflite file — check the output for errors.",
                file=sys.stderr,
            )
            sys.exit(1)
        # Pick the float32 variant if multiple files were emitted.
        tflite_src = next(
            (f for f in candidates if "float32" in f.name),
            candidates[0],
        )
        shutil.copy2(tflite_src, _TFLITE_PATH)

    print(f"Written:  {_TFLITE_PATH}  ({_TFLITE_PATH.stat().st_size // 1024} KB)")
    print("Run `uv run python scripts/benchmark_model.py --model yolo --units npu gpu` to verify.")


if __name__ == "__main__":
    convert()
