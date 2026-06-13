#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "moment-to-action",
# ]
#
# [tool.uv.sources]
# moment-to-action = { path = ".." }
# ///
"""Benchmark detection models on a COCO val2017 subset.

Downloads ~N images from COCO val2017, runs yolo_v8 / rf_detr / rtm_det on
each image for each backend (cpu, gpu, npu) x 3 load/infer/unload cycles, and
reports per-image latency + AP50 accuracy.

Usage:
    uv run python scripts/benchmark_detectors.py [--n-images 50] [--output benchmark_results.csv]

Requires QAI_HUB_API_TOKEN or QAI_HUB_API_KEY for npu backend; npu is skipped
gracefully when the environment is not configured or the backend is unavailable.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import httpx
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Iterator

from moment_to_action.config import load_config
from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.metrics import MetricsCollector, SpanType
from moment_to_action.models import MODEL_REGISTRY, ModelID, ModelManager
from moment_to_action.paths import PathManager
from moment_to_action.qairt import QairtSDKManager

console = Console()

_COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
_COCO_IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/val2017/{filename}"
_CACHE_DIR = Path.home() / ".cache" / "m2a-benchmark" / "coco"

# (ModelID, display name, registry variant key).  This benchmark runs ON the
# QCS6490 device, so every backend (cpu/gpu/npu) loads the same qcs DLC variant;
# resolve_backend_artifact picks the NPU context binary for npu and the portable
# DLC for cpu/gpu.  rf_detr/rtmdet are DLC-only (no NPU binary); detectron2 has
# two precision variants benchmarked separately.
_MODEL_CONFIGS: list[tuple[ModelID, str, str]] = [
    (ModelID.YOLO_V8, "yolo_v8", "qcs6490"),
    (ModelID.RF_DETR, "rf_detr", "qcs6490"),
    (ModelID.RTM_DET, "rtm_det", "qcs6490"),
    (ModelID.DETECTRON2, "detectron2_w8a16", "qcs6490_w8a16"),
    (ModelID.DETECTRON2, "detectron2_w8a8", "qcs6490_w8a8"),
]

_BACKENDS: list[tuple[str, ComputeUnit]] = [
    ("cpu", ComputeUnit.CPU),
    ("gpu", ComputeUnit.GPU),
    ("npu", ComputeUnit.NPU),
]

_N_CYCLES = 3
_IOU_THRESHOLD_AP50 = 0.5

# Toggled by --hw-metrics.  When False, traces record timing only (no power /
# frequency sampling) — avoids per-sample sensor-read warnings on boards whose
# power sysfs path is absent.  Set in main().
_HW_METRICS = False

# Sub progress bar tracking per-image progress within the current run.  Set in
# main() so the nested _run_cycle can advance it without threading it through
# every call.
_IMG_PROGRESS: Progress | None = None
_IMG_TASK: TaskID | None = None


def _advance_image() -> None:
    """Advance the per-image sub progress bar by one, if it is active."""
    if _IMG_PROGRESS is not None and _IMG_TASK is not None:
        _IMG_PROGRESS.advance(_IMG_TASK)


# ---------------------------------------------------------------------------
# Native log suppression
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _silence_native_output() -> Iterator[None]:
    """Redirect OS-level stdout+stderr to /dev/null for the duration of the block.

    The QAIRT runtime emits C++ logger chatter (e.g. "Profile Logger with name =
    defaultKey doesn't exist!") straight to file descriptors 1/2, bypassing
    Python's logging and corrupting the rich progress bar.  Wrapping the QAIRT
    calls (load/run/unload) in this redirects those fds to /dev/null and restores
    them afterwards, so only Python-level output reaches the terminal.

    Yields:
        None.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(devnull)
        os.close(saved[0])
        os.close(saved[1])


# ---------------------------------------------------------------------------
# COCO download helpers
# ---------------------------------------------------------------------------


def _ensure_annotations(cache_dir: Path) -> dict:
    """Download and cache COCO val2017 instance annotations.

    Downloads the annotations ZIP if not already cached, extracts
    ``instances_val2017.json``, and returns the parsed annotation dict.

    Args:
        cache_dir: Local cache directory for COCO data.

    Returns:
        Parsed COCO annotation dict with ``images`` and ``annotations`` keys.
    """
    ann_path = cache_dir / "annotations" / "instances_val2017.json"
    if ann_path.exists():
        with ann_path.open() as f:
            return json.load(f)

    import zipfile  # noqa: PLC0415

    zip_path = cache_dir / "annotations_trainval2017.zip"
    cache_dir.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Downloading COCO annotations (~241 MB)…[/cyan]")
    with httpx.stream("GET", _COCO_ANNOTATIONS_URL, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("annotations", total=total or None)
            with zip_path.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    progress.advance(task, len(chunk))

    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("annotations/instances_val2017.json", path=str(cache_dir))

    zip_path.unlink(missing_ok=True)
    with ann_path.open() as f:
        return json.load(f)


def _download_images(image_infos: list[dict], cache_dir: Path) -> list[Path]:
    """Download COCO images that are not already cached.

    Args:
        image_infos: List of COCO image info dicts (each with ``file_name``).
        cache_dir: Local cache root; images go into ``<cache_dir>/val2017/``.

    Returns:
        List of local paths, one per input image info entry.
    """
    img_dir = cache_dir / "val2017"
    img_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    to_download: list[tuple[int, dict]] = []

    for i, info in enumerate(image_infos):
        dest = img_dir / info["file_name"]
        paths.append(dest)
        if not dest.exists():
            to_download.append((i, info))

    if not to_download:
        return paths

    console.print(f"[cyan]Downloading {len(to_download)} COCO images…[/cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("images", total=len(to_download))
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            for _, info in to_download:
                url = _COCO_IMAGE_URL_TEMPLATE.format(filename=info["file_name"])
                dest = img_dir / info["file_name"]
                resp = client.get(url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                progress.advance(task)

    return paths


# ---------------------------------------------------------------------------
# AP50 computation (pure numpy)
# ---------------------------------------------------------------------------


def _iou(box_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and an array of boxes.

    Args:
        box_a: Shape ``(4,)`` array ``[x1, y1, x2, y2]``.
        boxes_b: Shape ``(N, 4)`` array.

    Returns:
        IoU values, shape ``(N,)``.
    """
    x1 = np.maximum(box_a[0], boxes_b[:, 0])
    y1 = np.maximum(box_a[1], boxes_b[:, 1])
    x2 = np.minimum(box_a[2], boxes_b[:, 2])
    y2 = np.minimum(box_a[3], boxes_b[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def _ap50(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
) -> float:
    """Compute AP50 for a single image (IoU threshold 0.5).

    Args:
        pred_boxes: ``(N, 4)`` predicted boxes ``[x1, y1, x2, y2]``.
        pred_scores: ``(N,)`` confidence scores.
        gt_boxes: ``(M, 4)`` ground-truth boxes ``[x1, y1, x2, y2]``.

    Returns:
        Average precision at IoU≥0.5.
    """
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0
    if len(pred_boxes) == 0:
        return 0.0

    order = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[order]

    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    for i, pb in enumerate(pred_boxes):
        ious = _iou(pb, gt_boxes)
        best = int(np.argmax(ious))
        if ious[best] >= _IOU_THRESHOLD_AP50 and not matched[best]:
            tp[i] = 1
            matched[best] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    n_gt = float(len(gt_boxes))
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)
    recall = cum_tp / n_gt

    # Interpolated AP
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall, strict=True):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _run_benchmark(
    manager: ModelManager,
    model_id: ModelID,
    variant: str,
    model_name: str,
    backend_name: str,
    unit: ComputeUnit,
    images: list[np.ndarray],
    gt_by_image: list[list[list[float]]],
) -> list[dict]:
    """Run one (model, backend, N_CYCLES) benchmark and return per-row results.

    Each row covers one (model, backend, image_id, run) combination.  Models are
    resolved and downloaded (if necessary) through ``manager.get_model``.  Each
    load/infer/unload cycle is wrapped in its own metrics trace, with on-device
    hardware sampling driven by the run's :class:`ComputeBackend`.

    Args:
        manager: ModelManager used to resolve/download/instantiate the model.
        model_id: Model to benchmark.
        variant: Registry variant key to load (qcs DLC variant on-device).
        model_name: Human-readable model name for output rows.
        backend_name: Backend name string for output rows.
        unit: :class:`~moment_to_action.hardware.ComputeUnit` to use.
        images: List of BGR uint8 frames.
        gt_by_image: List of GT box lists per image ``[[x1,y1,x2,y2], …]``.

    Returns:
        List of dicts, each representing one CSV row.
    """
    rows: list[dict] = []

    # --- construct backend; skip if this compute unit is unsupported on this device ---
    try:
        backend = ComputeBackend(unit)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"  [yellow]Skip {model_name}/{backend_name}: backend unavailable ({exc})[/yellow]"
        )
        return rows
    if backend.active_unit != unit:
        console.print(
            f"  [yellow]Skip {model_name}/{backend_name}: {backend_name} not supported "
            f"(fell back to {backend.active_unit.name.lower()}).[/yellow]"
        )
        return rows

    try:
        model = manager.get_model(model_id, variant=variant, unit=unit)
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [yellow]Skip {model_name}/{backend_name} ({variant}): {exc}[/yellow]")
        return rows

    metrics = MetricsCollector(backend if _HW_METRICS else None)

    for cycle in range(1, _N_CYCLES + 1):
        # One trace per load/infer/unload cycle (drives hardware sampling).
        with metrics.start_trace():
            cycle_rows = _run_cycle(
                model=model,
                backend=backend,
                model_name=model_name,
                backend_name=backend_name,
                cycle=cycle,
                images=images,
                gt_by_image=gt_by_image,
                metrics=metrics,
            )
        rows.extend(cycle_rows)

    report = metrics.report()
    if report.traces:
        console.print(
            f"  [dim]{model_name}/{backend_name}: {len(report.traces)} traces "
            f"({len(report.slow_traces)} over budget)[/dim]"
        )
    return rows


def _run_cycle(
    model: object,
    backend: ComputeBackend,
    model_name: str,
    backend_name: str,
    cycle: int,
    images: list[np.ndarray],
    gt_by_image: list[list[list[float]]],
    metrics: MetricsCollector,
) -> list[dict]:
    """Run one load → per-image infer → unload cycle inside an active trace.

    Args:
        model: Unloaded detection model instance.
        backend: ComputeBackend to load the model onto.
        model_name: Model name for output rows.
        backend_name: Backend name for output rows.
        cycle: Cycle index (1-based).
        images: List of BGR uint8 frames.
        gt_by_image: Ground-truth boxes per image.
        metrics: Active MetricsCollector (a trace must be open).

    Returns:
        Rows produced this cycle (empty if load failed).
    """
    # --- load (abort cycle on failure) ---
    t_load_start = time.perf_counter_ns()
    try:
        with (
            metrics.start_span(SpanType.MODEL_LOAD, f"{model_name}.{backend_name}.load"),
            _silence_native_output(),
        ):
            model.load(backend)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"  [yellow]Load failed {model_name}/{backend_name} cycle {cycle}: {exc}[/yellow]"
        )
        _safe_unload(model)
        return []
    load_ms = (time.perf_counter_ns() - t_load_start) / 1e6

    cycle_rows: list[dict] = []
    for img_idx, (frame, gt_boxes_raw) in enumerate(zip(images, gt_by_image, strict=True)):
        row = _process_image(
            model=model,
            frame=frame,
            gt_boxes_raw=gt_boxes_raw,
            model_name=model_name,
            backend_name=backend_name,
            img_idx=img_idx,
            cycle=cycle,
            load_ms=load_ms,
            metrics=metrics,
        )
        if row is not None:
            cycle_rows.append(row)
        _advance_image()

    # --- unload (best-effort) and backfill timing on this cycle's rows ---
    t_unload = time.perf_counter_ns()
    _safe_unload(model, metrics=metrics, span_name=f"{model_name}.{backend_name}.unload")
    unload_ms = (time.perf_counter_ns() - t_unload) / 1e6
    for row in cycle_rows:
        row["unload_ms"] = round(unload_ms, 3)
    return cycle_rows


def _safe_unload(
    model: object,
    metrics: MetricsCollector | None = None,
    span_name: str = "unload",
) -> None:
    """Unload a model, swallowing any error so the benchmark can continue.

    Args:
        model: The model to unload (must have an ``unload()`` method).
        metrics: Optional collector to wrap the unload in a span.
        span_name: Span name to use when ``metrics`` is provided.
    """
    try:
        if metrics is not None:
            with (
                metrics.start_span(SpanType.MODEL_UNLOAD, span_name),
                _silence_native_output(),
            ):
                model.unload()  # type: ignore[attr-defined]
        else:
            with _silence_native_output():
                model.unload()  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [yellow]Unload failed: {exc}[/yellow]")


def _process_image(  # noqa: PLR0913
    model: object,
    frame: np.ndarray,
    gt_boxes_raw: list[list[float]],
    model_name: str,
    backend_name: str,
    img_idx: int,
    cycle: int,
    load_ms: float,
    metrics: MetricsCollector,
) -> dict | None:
    """Run prepare/infer/post/AP50 for one image, returning a row or None on error.

    Any exception is caught and logged so the benchmark continues with the next
    image rather than aborting the whole run.

    Args:
        model: Loaded detection model.
        frame: BGR uint8 image.
        gt_boxes_raw: Ground-truth boxes ``[[x1,y1,x2,y2], …]`` for this image.
        model_name: Model name for the output row.
        backend_name: Backend name for the output row.
        img_idx: Index of this image.
        cycle: Current benchmark cycle.
        load_ms: Load latency recorded for this cycle.
        metrics: MetricsCollector for spans.

    Returns:
        A CSV row dict, or ``None`` if inference failed for this image.
    """
    try:
        t_pre = time.perf_counter_ns()
        with metrics.start_span(SpanType.MODEL_PREPROCESS, f"{model_name}.preproc"):
            prepared = model.prepare(frame)  # type: ignore[attr-defined]
        pre_ms = (time.perf_counter_ns() - t_pre) / 1e6

        t_inf = time.perf_counter_ns()
        with (
            metrics.start_span(SpanType.MODEL_INFERENCE, f"{model_name}.infer"),
            _silence_native_output(),
        ):
            raw = model.run(prepared)  # type: ignore[attr-defined]
        inf_ms = (time.perf_counter_ns() - t_inf) / 1e6

        t_post = time.perf_counter_ns()
        with metrics.start_span(SpanType.MODEL_POST_PROCESS, f"{model_name}.post"):
            detections = model.post_proc(raw)  # type: ignore[attr-defined]
        post_ms = (time.perf_counter_ns() - t_post) / 1e6

        pred_boxes = np.array(
            [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections],
            dtype=np.float32,
        ).reshape(-1, 4)
        pred_scores = np.array([d.confidence for d in detections], dtype=np.float32)
        gt_boxes = np.array(gt_boxes_raw, dtype=np.float32).reshape(-1, 4)
        ap = _ap50(pred_boxes, pred_scores, gt_boxes)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"  [yellow]Image {img_idx} failed {model_name}/{backend_name} "
            f"cycle {cycle}: {exc}[/yellow]"
        )
        return None

    return {
        "model": model_name,
        "backend": backend_name,
        "image_idx": img_idx,
        "run": cycle,
        "load_ms": round(load_ms, 3),
        "preproc_ms": round(pre_ms, 3),
        "infer_ms": round(inf_ms, 3),
        "postproc_ms": round(post_ms, 3),
        "unload_ms": 0.0,  # filled by caller after unload
        "ap50": round(ap, 4),
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(all_rows: list[dict]) -> None:
    """Print a rich summary table with averages per (model, backend).

    Args:
        all_rows: All result rows from the benchmark.
    """
    # Group by (model, backend)
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in all_rows:
        key = (row["model"], row["backend"])
        groups.setdefault(key, []).append(row)

    table = Table(title="Detector Benchmark Summary", show_lines=True)
    table.add_column("Model", style="bold cyan")
    table.add_column("Backend", style="bold magenta")
    table.add_column("Load (ms)", justify="right")
    table.add_column("Preproc (ms)", justify="right")
    table.add_column("Infer (ms)", justify="right")
    table.add_column("Postproc (ms)", justify="right")
    table.add_column("Unload (ms)", justify="right")
    table.add_column("AP50", justify="right", style="bold green")

    for (model_name, backend_name), rows in sorted(groups.items()):

        def avg(key: str, _rows: list[dict] = rows) -> str:
            vals = [r[key] for r in _rows]
            return f"{np.mean(vals):.2f}"

        table.add_row(
            model_name,
            backend_name,
            avg("load_ms"),
            avg("preproc_ms"),
            avg("infer_ms"),
            avg("postproc_ms"),
            avg("unload_ms"),
            avg("ap50"),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark detection models on COCO val2017 subset."
    )
    parser.add_argument("--n-images", type=int, default=50, help="Number of COCO images to use.")
    parser.add_argument(
        "--output",
        default="benchmark_results.csv",
        help="Output CSV path (default: benchmark_results.csv).",
    )
    parser.add_argument(
        "--hw-metrics",
        action="store_true",
        help="Sample on-device power/frequency during traces (off by default; "
        "enable only where the power sysfs path exists, else it logs per-sample warnings).",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model display names to run (default: all). "
        "E.g. --models detectron2_w8a16,detectron2_w8a8",
    )
    parser.add_argument(
        "--backends",
        default=None,
        help="Comma-separated backends to run: cpu,gpu,npu (default: all).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge results into the existing --output CSV: rows for the "
        "(model, backend) pairs re-run are replaced; all others are kept. "
        "Use with --models/--backends to re-run just one combo after a fix.",
    )
    return parser.parse_args()


# CSV columns and their types, used when merging an existing results file.
_CSV_FLOAT_FIELDS = ("load_ms", "preproc_ms", "infer_ms", "postproc_ms", "unload_ms", "ap50")
_CSV_INT_FIELDS = ("image_idx", "run")


def _read_existing_csv(path: Path) -> list[dict]:
    """Read an existing results CSV, coercing numeric columns back to numbers.

    Args:
        path: Path to a CSV previously written by :func:`_write_csv`.

    Returns:
        List of row dicts with numeric fields as ``float``/``int`` (so they can be
        averaged alongside freshly produced rows).
    """
    rows: list[dict] = []
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            row = dict(raw)
            for k in _CSV_FLOAT_FIELDS:
                row[k] = float(row[k])
            for k in _CSV_INT_FIELDS:
                row[k] = int(row[k])
            rows.append(row)
    return rows


def _merge_rows(
    existing_path: Path,
    new_rows: list[dict],
    rerun_pairs: set[tuple[str, str]],
) -> list[dict]:
    """Merge ``new_rows`` into an existing CSV, replacing the re-run pairs.

    Rows in the existing file whose ``(model, backend)`` is in ``rerun_pairs`` are
    dropped (they were just re-run); everything else is kept and the new rows are
    appended.

    Args:
        existing_path: Path to the existing results CSV (may not exist).
        new_rows: Freshly produced rows from this run.
        rerun_pairs: ``(model, backend)`` pairs that were attempted this run.

    Returns:
        The merged row list.
    """
    if not existing_path.exists():
        return new_rows
    existing = _read_existing_csv(existing_path)
    kept = [r for r in existing if (r["model"], r["backend"]) not in rerun_pairs]
    return kept + new_rows


def _load_coco_images(
    n_images: int,
) -> tuple[list[np.ndarray], list[list[list[float]]]]:
    """Download and load COCO val2017 images with ground-truth boxes.

    Args:
        n_images: Number of images to load.

    Returns:
        Tuple of ``(images, gt_boxes_list)`` where ``images`` is a list of BGR
        uint8 frames and ``gt_boxes_list`` is the corresponding list of
        ``[[x1, y1, x2, y2], …]`` ground-truth boxes per image.
    """
    ann = _ensure_annotations(_CACHE_DIR)
    image_infos = ann["images"][:n_images]
    img_paths = _download_images(image_infos, _CACHE_DIR)

    gt_by_id: dict[int, list[list[float]]] = {info["id"]: [] for info in image_infos}
    for aobj in ann["annotations"]:
        if aobj["image_id"] in gt_by_id:
            x, y, w, h = aobj["bbox"]
            gt_by_id[aobj["image_id"]].append([x, y, x + w, y + h])

    console.print("[cyan]Loading images…[/cyan]")
    images: list[np.ndarray] = []
    gt_boxes_list: list[list[list[float]]] = []
    for info, img_path in zip(image_infos, img_paths, strict=True):
        frame = cv2.imread(str(img_path))
        if frame is None:
            console.print(f"  [red]Failed to load {img_path}, skipping.[/red]")
            continue
        images.append(frame)
        gt_boxes_list.append(gt_by_id[info["id"]])

    return images, gt_boxes_list


def _write_csv(rows: list[dict], output_path: Path) -> None:
    """Write benchmark result rows to a CSV file.

    Args:
        rows: List of result dicts (one per model/backend/image/run).
        output_path: Destination CSV path.
    """
    fieldnames = [
        "model",
        "backend",
        "image_idx",
        "run",
        "load_ms",
        "preproc_ms",
        "infer_ms",
        "postproc_ms",
        "unload_ms",
        "ap50",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _configure_qairt() -> None:
    """Set up the QAIRT SDK environment (QAIRT_SDK_ROOT etc.) for DLC loading.

    Mirrors the ``m2a`` CLI root callback: without this, ``load_model_dlc``
    raises "QAIRT SDK is not available" even when the SDK is installed, because
    the environment variables are never exported into this process.
    """
    path_manager = PathManager()
    config = load_config(path_manager.app_config_file)
    if config.qairt_sdk_path is None:
        console.print(
            "  [yellow]QAIRT SDK path not configured — DLC backends may be unavailable.[/yellow]"
        )
        return
    try:
        QairtSDKManager.from_app_config(config, path_manager).configure_env()
    except RuntimeError as exc:
        console.print(f"  [yellow]QAIRT env setup failed: {exc}[/yellow]")


def main() -> None:  # noqa: PLR0915
    """Entry point for the benchmark script."""
    global _HW_METRICS, _IMG_PROGRESS, _IMG_TASK  # noqa: PLW0603
    args = _parse_args()
    n_images: int = args.n_images
    output_path = Path(args.output)
    _HW_METRICS = bool(args.hw_metrics)

    # Optional subset filters (for re-running a single model/backend after a fix).
    model_filter = set(args.models.split(",")) if args.models else None
    backend_filter = set(args.backends.split(",")) if args.backends else None
    configs = [c for c in _MODEL_CONFIGS if model_filter is None or c[1] in model_filter]
    backends = [b for b in _BACKENDS if backend_filter is None or b[0] in backend_filter]

    console.rule("[bold]M2A Detector Benchmark[/bold]")
    console.print(f"  images : {n_images}")
    console.print(f"  cycles : {_N_CYCLES}")
    console.print(f"  output : {output_path}")
    console.print(f"  models : {', '.join(c[1] for c in configs)}")
    console.print(f"  backend: {', '.join(b[0] for b in backends)}")
    if args.merge:
        console.print("  merge  : on (re-run pairs replace existing rows)")
    console.print()

    if not configs or not backends:
        console.print("[red]No model/backend selected by filters. Exiting.[/red]")
        sys.exit(1)

    _configure_qairt()

    images, gt_boxes_list = _load_coco_images(n_images)

    if not images:
        console.print("[red]No images loaded. Exiting.[/red]")
        sys.exit(1)

    console.print(f"  Loaded {len(images)} images.\n")

    # 2. Run benchmarks (each run builds its own MetricsCollector + per-cycle trace)
    manager = ModelManager(PathManager())
    all_rows: list[dict] = []
    rerun_pairs: set[tuple[str, str]] = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        total_runs = len(configs) * len(backends)
        task = progress.add_task("benchmarking", total=total_runs)
        img_task = progress.add_task("images", total=len(images) * _N_CYCLES)
        _IMG_PROGRESS, _IMG_TASK = progress, img_task

        for model_id, model_name, variant in configs:
            info = MODEL_REGISTRY.get(model_id)
            if info is None or variant not in info.variants:
                console.print(
                    f"  [yellow]{model_name} ({variant}) not in registry, skipping.[/yellow]"
                )
                progress.advance(task, len(backends))
                continue

            for backend_name, unit in backends:
                rerun_pairs.add((model_name, backend_name))
                progress.update(task, description=f"{model_name}/{backend_name}")
                progress.reset(
                    img_task,
                    total=len(images) * _N_CYCLES,
                    description=f"  {model_name}/{backend_name} imgs",
                )
                try:
                    rows = _run_benchmark(
                        manager=manager,
                        model_id=model_id,
                        variant=variant,
                        model_name=model_name,
                        backend_name=backend_name,
                        unit=unit,
                        images=images,
                        gt_by_image=gt_boxes_list,
                    )
                    all_rows.extend(rows)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [red]Run {model_name}/{backend_name} crashed: {exc}[/red]")
                progress.advance(task)

    # 3. Merge with existing results (if requested) and write CSV
    if args.merge:
        all_rows = _merge_rows(output_path, all_rows, rerun_pairs)

    if all_rows:
        _write_csv(all_rows, output_path)
        console.print(f"\n[green]Results written to {output_path}[/green]")
    else:
        console.print("[yellow]No results produced.[/yellow]")

    # 4. Print summary table (full merged picture)
    console.print()
    _print_summary(all_rows)


if __name__ == "__main__":
    main()
