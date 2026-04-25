"""test_all_drivers.py — YOLOv12 cross-backend latency benchmark.

Compares three Qualcomm compute engines on the same YOLOv12 architecture:

    CPU  — ONNX FP32  (ground-truth precision, slowest)
    NPU  — ONNX QDQ INT8  (HTP backend; quantized in this script on first run)
    GPU  — ONNX FP16  (Adreno backend; converted from FP32 on first run)

Model pipeline
--------------
    1. ModelManager downloads yolo12n.onnx (FP32) from HF on first run.
    2. FP16 variant is derived from FP32 via onnxconverter-common (keep_io_types=True).
    3. QDQ INT8 variant is derived from the pre-processed FP32 via ORT static quantization,
         calibrated with a subset of real COCO images pulled through CocoDataset.

All three variants are cached next to the FP32 source so subsequent runs are instant.

Usage
-----
    uv run test_all_drivers.py
    uv run test_all_drivers.py /path/to/yolo12n.onnx   # skip download, use this FP32 file.
"""

from __future__ import annotations

# Standard library imports
import sys
import time
from pathlib import Path

# Third-party imports
import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)

# Local application imports
from moment_to_action.benchmark import CocoDataset
from moment_to_action.models import ModelManager
from moment_to_action.models._types import ModelID

# ── N calibration images used when building the QDQ model ────────────────────
N_CALIB = 128
# YOLO12n native input size
YOLO_INPUT_HW = (640, 640)


# ── Paths ─────────────────────────────────────────────────────────────────────


def _resolve_paths(fp32_override: str | None) -> tuple[Path, Path, Path]:
    """Return (fp32_path, fp16_path, qdq_path).

    If an explicit FP32 path is given on the CLI, use it directly.
    Otherwise, ask ModelManager to download/cache the FP32 model from HF.
    The FP16 and QDQ variants sit alongside the FP32 file.
    """
    if fp32_override:
        fp32 = Path(fp32_override).resolve()
    else:
        manager = ModelManager()
        fp32 = manager.get_path(ModelID.YOLO_V12_N)

    fp16 = fp32.with_name(fp32.stem + "_fp16.onnx")
    qdq = fp32.with_name(fp32.stem + "_qdq.onnx")
    return fp32, fp16, qdq


# ── FP16 derivation ───────────────────────────────────────────────────────────


def ensure_fp16(fp32: Path, fp16: Path) -> Path:
    """Convert FP32 → FP16 if not already cached.

    keep_io_types=True keeps the graph I/O at float32 so every backend
    receives the same np.float32 dummy input — no per-backend dtype handling.
    """
    if fp16.exists():
        return fp16

    print(f"[fp16] Converting {fp32.name} → {fp16.name} …")
    model_fp16 = float16.convert_float_to_float16(
        onnx.load(str(fp32)),
        keep_io_types=True,
    )
    onnx.save(model_fp16, str(fp16))
    print(f"[fp16] Saved: {fp16}\n")
    return fp16


# ── QDQ derivation ────────────────────────────────────────────────────────────


class _CocoCalibReader(CalibrationDataReader):
    """Feeds pre-processed COCO images as calibration tensors.

    Uses CocoDataset to load real images so that activation ranges are
    computed on representative data rather than random noise.
    """

    def __init__(
        self,
        input_name: str,
        n_samples: int = N_CALIB,
        input_hw: tuple[int, int] = YOLO_INPUT_HW,
    ) -> None:
        import cv2  # optional; only needed during QDQ generation

        self._input_name = input_name
        dataset = CocoDataset(n_images=n_samples)
        images = dataset.images()[:n_samples]

        self._samples: list[np.ndarray] = []
        h, w = input_hw

        for img_meta in images:
            # CocoDataset.images() returns dicts with a 'file_name' or 'path' key.
            # Adjust the key name to match whatever your CocoDataset actually returns.
            img_path = img_meta.get("file_name") or img_meta.get("path", "")  # type: ignore[attr-defined]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, (w, h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = img.astype(np.float32) / 255.0  # [0,1]
            tensor = np.transpose(tensor, (2, 0, 1))  # HWC → CHW
            tensor = np.expand_dims(tensor, 0)  # CHW → 1CHW
            self._samples.append(tensor)

        if not self._samples:
            msg = (
                "CocoDataset returned no loadable images for QDQ calibration. "
                "Check that COCO val images are downloaded and paths are correct."
            )
            raise RuntimeError(msg)

        self._iter = iter(self._samples)

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            return {self._input_name: next(self._iter)}
        except StopIteration:
            return None


def ensure_qdq(fp32: Path, qdq: Path) -> Path:
    """Derive a QDQ INT8 model from the FP32 source if not already cached.

    Steps:
      1. quant_pre_process — shape inference + graph optimisation (ORT recommended)
      2. quantize_static with QDQ format + COCO calibration data
    """
    if qdq.exists():
        return qdq

    prep = fp32.with_name(fp32.stem + "_prep.onnx")

    print(f"[qdq] Pre-processing {fp32.name} …")
    quant_pre_process(
        input_model_path=str(fp32),
        output_model_path=str(prep),
        auto_merge=True,
        save_as_external_data=False,
    )

    # Infer input name from the pre-processed model for the calibration reader
    input_name = onnx.load(str(prep)).graph.input[0].name

    print(f"[qdq] Calibrating on {N_CALIB} COCO images …")
    reader = _CocoCalibReader(input_name=input_name, n_samples=N_CALIB)

    quantize_static(
        model_input=str(prep),
        model_output=str(qdq),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        # per_channel improves accuracy but may fail on some QNN HTP op configs.
        # Start with False; enable if NPU accuracy is acceptable.
        per_channel=False,
    )

    # Clean up the intermediate pre-processed file
    prep.unlink(missing_ok=True)

    print(f"[qdq] Saved: {qdq}\n")
    return qdq


# ── Inference helpers ─────────────────────────────────────────────────────────


def _make_dummy(session: ort.InferenceSession) -> np.ndarray:
    shape = [1 if isinstance(d, str) or d is None else d for d in session.get_inputs()[0].shape]
    rng = np.random.default_rng()
    return rng.random(shape, dtype=np.float32)


def run_backend(
    label: str,
    providers: list,
    model_path: Path | str,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> dict | None:
    """Load a session, warm up, collect latency stats. Returns None on failure."""
    w = 58
    print(f"\n{'─' * w}")
    print(f"  {label}")
    print(f"{'─' * w}")

    model_path_str = str(model_path)
    if not Path(model_path_str).exists():
        print(f"  SKIPPED — model not found: {model_path_str}")
        return None

    try:
        so = ort.SessionOptions()
        sess = ort.InferenceSession(model_path_str, sess_options=so, providers=providers)

        print(f"  Active providers : {sess.get_providers()}")
        print(f"  Model            : {Path(model_path_str).name}")

        inp_name = sess.get_inputs()[0].name
        dummy = _make_dummy(sess)

        for _ in range(n_warmup):
            sess.run(None, {inp_name: dummy})

        times: list[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            sess.run(None, {inp_name: dummy})
            times.append((time.perf_counter() - t0) * 1_000)

        ts = sorted(times)
        stats = {
            "mean": sum(times) / len(times),
            "p50": ts[len(ts) // 2],
            "min": ts[0],
            "max": ts[-1],
        }
        print(
            f"  Latency (ms)     : "
            f"mean={stats['mean']:.2f}  p50={stats['p50']:.2f}  "
            f"min={stats['min']:.2f}  max={stats['max']:.2f}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {exc}")
        return None
    else:
        return stats


# ── Provider configs ──────────────────────────────────────────────────────────

CPU_PROVIDERS = ["CPUExecutionProvider"]

NPU_PROVIDERS = [
    (
        "QNNExecutionProvider",
        {
            "backend_path": "/usr/lib/libQnnHtp.so",
            "profiling_level": "basic",
            "profiling_file_path": "qnn_profile_npu.csv",
        },
    ),
    "CPUExecutionProvider",
]

GPU_PROVIDERS = [
    (
        "QNNExecutionProvider",
        {
            "backend_path": "/usr/lib/libQnnGpu.so",
            "profiling_level": "basic",
            "profiling_file_path": "qnn_profile_gpu.csv",
        },
    ),
    "CPUExecutionProvider",
]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fp32_model, fp16_model, qdq_model = _resolve_paths(sys.argv[1] if len(sys.argv) > 1 else None)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║       YOLOv12 Cross-Backend Latency Benchmark            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  FP32 source  : {fp32_model}")
    print(f"  FP16 (GPU)   : {fp16_model}  [auto-derived]")
    print(f"  QDQ  (NPU)   : {qdq_model}   [auto-derived]\n")

    # --- Derive variants (only runs once; subsequent runs load from cache) ---
    try:
        fp16_model = ensure_fp16(fp32_model, fp16_model)
    except Exception as exc:  # noqa: BLE001
        print(f"[fp16] FAILED: {exc}")
        fp16_model = None  # type: ignore[assignment]

    try:
        qdq_model = ensure_qdq(fp32_model, qdq_model)
    except Exception as exc:  # noqa: BLE001
        print(f"[qdq] FAILED: {exc}")
        qdq_model = None  # type: ignore[assignment]

    # --- Run benchmarks ---
    results: dict[str, dict | None] = {}

    results["CPU  (FP32)"] = run_backend(
        "CPU — FP32  [ground truth]",
        CPU_PROVIDERS,
        fp32_model,
    )
    results["NPU  (QDQ INT8)"] = run_backend(
        "NPU — QNN HTP / QDQ INT8",
        NPU_PROVIDERS,
        qdq_model or "",
    )
    results["GPU  (FP16)"] = run_backend(
        "GPU — QNN Adreno / FP16",
        GPU_PROVIDERS,
        fp16_model or "",
    )

    # --- Summary ---
    w = 58
    print(f"\n\n{'═' * w}")
    print(f"  {'SUMMARY':^{w - 4}}")
    print(f"{'═' * w}")
    print(f"  {'Backend':<22} {'Status':<8} {'Mean ms':>8}  {'p50 ms':>8}")
    print(f"  {'─' * 20} {'─' * 6} {'─' * 8}  {'─' * 8}")
    for name, stats in results.items():
        if stats:
            print(f"  {name:<22} {'OK':<8} {stats['mean']:>8.2f}  {stats['p50']:>8.2f}")
        else:
            print(f"  {name:<22} {'FAILED':<8} {'—':>8}  {'—':>8}")
    print(f"{'═' * w}\n")
