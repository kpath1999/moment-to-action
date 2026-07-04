#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "moment-to-action",
# ]
#
# [tool.uv.sources]
# moment-to-action = { path = "..", editable = true }
# ///
"""Benchmark detection models on a COCO val2017 subset.

Downloads ~N images from COCO val2017, runs yolo_v8 / rf_detr / rtm_det on
each image for each backend (cpu, gpu, npu) x 3 load/infer/unload cycles, and
reports per-image AP50 accuracy.  All latency data is captured via
MetricsCollector and included in the JSON output (load, preproc, inference,
postproc, and unload spans with hardware resource samples).

Each (model, backend, cycle, image) is driven through a real
``Pipeline([ImageDetectionStage(model)])`` over a ``RawFrameMessage`` — this is
the same composition an on-device app would use, just fed COCO frames instead
of a live camera.

Usage:
    uv run python bench/benchmark_detectors.py [--n-images 50] [--output benchmark_results.json]

Requires QAI_HUB_API_TOKEN or QAI_HUB_API_KEY for npu backend; npu is skipped
gracefully when the environment is not configured or the backend is unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
from _common import BenchContext, build_context, console, silence_native_output, write_results
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

from moment_to_action.benchmarking import ap50
from moment_to_action.hardware import ComputeUnit
from moment_to_action.messages import DetectionMessage, RawFrameMessage
from moment_to_action.models import MODEL_REGISTRY, ModelID
from moment_to_action.pipeline import Pipeline
from moment_to_action.stages.image import ImageDetectionStage

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

# Sub progress bar tracking per-image progress within the current run.  Set in
# main() so the nested run helpers can advance it without threading it through
# every call.
_IMG_PROGRESS: Progress | None = None
_IMG_TASK: TaskID | None = None


def _advance_image() -> None:
    """Advance the per-image sub progress bar by one, if it is active."""
    if _IMG_PROGRESS is not None and _IMG_TASK is not None:
        _IMG_PROGRESS.advance(_IMG_TASK)


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
# Benchmark runner
# ---------------------------------------------------------------------------


def _run_benchmark(
    ctx: BenchContext,
    model_id: ModelID,
    variant: str,
    model_name: str,
    backend_name: str,
    unit: ComputeUnit,
    images: list[np.ndarray],
    gt_by_image: list[list[list[float]]],
) -> list[dict]:
    """Run one (model, backend, N_CYCLES) benchmark and return per-cycle result rows.

    Each row covers one (model, backend, cycle) combination and contains per-image
    AP50 values plus this cycle's trace JSON (load + preproc + infer + postproc +
    unload spans with hardware resource samples).

    Args:
        ctx: Shared bench context (one MetricsCollector/ModelManager for the run).
        model_id: Model to benchmark.
        variant: Registry variant key to load (qcs DLC variant on-device).
        model_name: Human-readable model name for output rows.
        backend_name: Backend name string for output rows.
        unit: Compute unit to benchmark.
        images: List of BGR uint8 frames.
        gt_by_image: List of GT box lists per image ``[[x1,y1,x2,y2], …]``.

    Returns:
        List of dicts, one per (model, backend, cycle).
    """
    rows: list[dict] = []

    if unit not in ctx.platform.supported_units:
        console.print(
            f"  [yellow]Skip {model_name}/{backend_name}: {backend_name} not available "
            f"on this platform.[/yellow]"
        )
        return rows

    try:
        model = ctx.manager.get_model(model_id, variant=variant, unit=unit)
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [yellow]Skip {model_name}/{backend_name} ({variant}): {exc}[/yellow]")
        return rows

    for cycle in range(1, _N_CYCLES + 1):
        with ctx.metrics.start_trace() as trace:
            cycle_row = _run_cycle(
                ctx=ctx,
                model=model,
                unit=unit,
                model_name=model_name,
                backend_name=backend_name,
                cycle=cycle,
                images=images,
                gt_by_image=gt_by_image,
            )
        if cycle_row is not None:
            cycle_row["trace"] = trace.json()
            rows.append(cycle_row)

    return rows


def _run_cycle(
    ctx: BenchContext,
    model: object,
    unit: ComputeUnit,
    model_name: str,
    backend_name: str,
    cycle: int,
    images: list[np.ndarray],
    gt_by_image: list[list[list[float]]],
) -> dict | None:
    """Run one load → per-image infer → unload cycle inside an active trace.

    Per-image inference goes through a real ``Pipeline([ImageDetectionStage(model)])``
    over a ``RawFrameMessage``, exercising the same composition an on-device app uses.
    All latency data is captured via MetricsCollector spans (MODEL_LOAD,
    MODEL_PREPROCESS, MODEL_INFERENCE, MODEL_POST_PROCESS, MODEL_UNLOAD, STAGE) on
    the collector the model was constructed with.

    Args:
        ctx: Shared bench context.
        model: Unloaded detection model instance.
        unit: Compute unit to use.
        model_name: Model name for output rows.
        backend_name: Backend name for output rows.
        cycle: Cycle index (1-based).
        images: List of BGR uint8 frames.
        gt_by_image: Ground-truth boxes per image.

    Returns:
        Cycle row dict, or ``None`` if load failed.
    """
    try:
        with silence_native_output():
            model.load(ctx.platform, unit)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"  [yellow]Load failed {model_name}/{backend_name} cycle {cycle}: {exc}[/yellow]"
        )
        _safe_unload(model)
        return None

    pipeline = Pipeline(
        [ImageDetectionStage(model, metrics=ctx.metrics)],  # type: ignore[arg-type]
        metrics=ctx.metrics,
    )

    image_results: list[dict] = []
    for img_idx, (frame, gt_boxes_raw) in enumerate(zip(images, gt_by_image, strict=True)):
        result = _process_image(
            pipeline=pipeline,
            frame=frame,
            gt_boxes_raw=gt_boxes_raw,
            model_name=model_name,
            backend_name=backend_name,
            img_idx=img_idx,
            cycle=cycle,
        )
        if result is not None:
            image_results.append(result)
        _advance_image()

    _safe_unload(model)

    return {
        "model": model_name,
        "backend": backend_name,
        "cycle": cycle,
        "image_results": image_results,
    }


def _safe_unload(model: object) -> None:
    """Unload a model, swallowing any error so the benchmark can continue.

    Args:
        model: The model to unload (must have an ``unload()`` method).
    """
    try:
        with silence_native_output():
            model.unload()  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [yellow]Unload failed: {exc}[/yellow]")


def _process_image(
    pipeline: Pipeline,
    frame: np.ndarray,
    gt_boxes_raw: list[list[float]],
    model_name: str,
    backend_name: str,
    img_idx: int,
    cycle: int,
) -> dict | None:
    """Run one frame through the detection pipeline and compute its AP50.

    Args:
        pipeline: ``Pipeline([ImageDetectionStage(model)])`` for the loaded model.
        frame: BGR uint8 image.
        gt_boxes_raw: Ground-truth boxes ``[[x1,y1,x2,y2], …]`` for this image.
        model_name: Model name for the output row.
        backend_name: Backend name for the output row.
        img_idx: Index of this image.
        cycle: Current benchmark cycle.

    Returns:
        A result dict with ``image_idx`` and ``ap50``, or ``None`` if inference failed.
    """
    try:
        source = iter([RawFrameMessage(frame=frame, timestamp=time.time())])
        with silence_native_output():
            (out,) = list(pipeline.run(source))
        assert isinstance(out, DetectionMessage)  # noqa: S101  # ImageDetectionStage guarantees this
        detections = out.detections

        pred_boxes = np.array(
            [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections],
            dtype=np.float32,
        ).reshape(-1, 4)
        pred_scores = np.array([d.confidence for d in detections], dtype=np.float32)
        gt_boxes = np.array(gt_boxes_raw, dtype=np.float32).reshape(-1, 4)
        ap = ap50(pred_boxes, pred_scores, gt_boxes)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"  [yellow]Image {img_idx} failed {model_name}/{backend_name} "
            f"cycle {cycle}: {exc}[/yellow]"
        )
        return None

    return {"image_idx": img_idx, "ap50": round(ap, 4)}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(all_rows: list[dict]) -> None:
    """Print a rich summary table with averages per (model, backend).

    Args:
        all_rows: All cycle result rows from the benchmark.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in all_rows:
        key = (row["model"], row["backend"])
        groups.setdefault(key, []).append(row)

    table = Table(title="Detector Benchmark Summary", show_lines=True)
    table.add_column("Model", style="bold cyan")
    table.add_column("Backend", style="bold magenta")
    table.add_column("Cycles", justify="right")
    table.add_column("Images", justify="right")
    table.add_column("AP50", justify="right", style="bold green")

    for (model_name, backend_name), rows in sorted(groups.items()):
        all_image_results = [r for row in rows for r in row["image_results"]]
        ap50_vals = [r["ap50"] for r in all_image_results]
        avg_ap50 = float(np.mean(ap50_vals)) if ap50_vals else float("nan")
        table.add_row(
            model_name,
            backend_name,
            str(len(rows)),
            str(len(all_image_results)),
            f"{avg_ap50:.4f}",
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
        default="benchmark_results.json",
        help="Output JSON path (default: benchmark_results.json).",
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
    return parser.parse_args()


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


def main() -> None:  # noqa: PLR0915
    """Entry point for the benchmark script."""
    global _IMG_PROGRESS, _IMG_TASK
    args = _parse_args()
    n_images: int = args.n_images
    output_path = Path(args.output)

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
    console.print()

    if not configs or not backends:
        console.print("[red]No model/backend selected by filters. Exiting.[/red]")
        sys.exit(1)

    ctx = build_context(qairt=True)
    images, gt_boxes_list = _load_coco_images(n_images)

    if not images:
        console.print("[red]No images loaded. Exiting.[/red]")
        sys.exit(1)

    console.print(f"  Loaded {len(images)} images.\n")

    all_rows: list[dict] = []

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
                progress.update(task, description=f"{model_name}/{backend_name}")
                progress.reset(
                    img_task,
                    total=len(images) * _N_CYCLES,
                    description=f"  {model_name}/{backend_name} imgs",
                )
                try:
                    rows = _run_benchmark(
                        ctx=ctx,
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

    if all_rows:
        write_results(all_rows, output_path, script="benchmark_detectors", entries_key="runs")
        console.print(f"\n[green]Results written to {output_path}[/green]")
    else:
        console.print("[yellow]No results produced.[/yellow]")

    console.print()
    _print_summary(all_rows)


if __name__ == "__main__":
    main()
