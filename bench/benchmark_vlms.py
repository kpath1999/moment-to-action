#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "moment-to-action",
#     "Pillow",
#     "opencv-python",
# ]
#
# [tool.uv.sources]
# moment-to-action = { path = "..", editable = true }
# ///
"""Benchmark VLM models on application-specific classification scenes with visual input.

Each scene (``bench/_scenes.py``, shared with ``benchmark_llms.py``) maps to one
of the five target applications. The VLM receives video frames directly —
rendered from the same scene bounding boxes used in ``benchmark_llms.py`` — and
answers the binary/multi-label question.

By default, synthetic frames are generated from scene bounding box data using
PIL (colored rectangles with label text on a gray canvas). If ``--video-dir``
is supplied and a file ``<dir>/<scene_name>.mp4`` exists for a scene, real
frames are sampled from that video instead.

Each (model, scene, cycle) is driven through a real
``Pipeline([VLMDescriptionStage(model, task, grammar=YES_NO_GRAMMAR), DecisionStage])``
over a ``VideoClipMessage`` of raw BGR frames — the same composition an
on-device app would use. Yes/no scenes get the grammar; PPE compliance scenes
(COMPLIANT/NON-COMPLIANT) run without it, scored on keyword recall only.

Accuracy metrics:
  - ``yn_correct``: bool — whether the extracted decision matched the expected label.
  - ``recall``: float in [0, 1] — keyword recall for classification keywords.

Timing metrics (streamed from ``MetricsCollector.timed_stream`` via ``VLMDescriptionStage``):
  - ``ttft_ms``: time from stream start to first token.
  - ``ttfyd_ms``: time from stream start to first yes/no decision.
  - ``mean_itl_ms``, ``std_itl_ms``: inter-token latency statistics.
  - ``inference_metrics``: llama.cpp-native timing fields from the stop chunk.

Usage:
    uv run python bench/benchmark_vlms.py [--n-cycles 3] [--output results.json]

Requires llama_server_path to be set in the M2A config (or pass --server-path).
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from _common import build_context, console, write_results
from _scenes import SCENES, Scene
from PIL import Image, ImageDraw, ImageFont
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

from moment_to_action.benchmarking import extract_load_unload_ms, recall
from moment_to_action.config import AppConfig
from moment_to_action.hardware import ComputeUnit
from moment_to_action.messages import DecisionMessage, GenerationMessage, VideoClipMessage
from moment_to_action.metrics import SpanType
from moment_to_action.models import MODEL_REGISTRY, ModelID
from moment_to_action.prompting import BENCHMARK_SYSTEM, YES_NO_GRAMMAR
from moment_to_action.stages.llm import DecisionStage
from moment_to_action.stages.vlm import VLMDescriptionStage
from moment_to_action.stages.vlm._encode import bgr_to_b64

if TYPE_CHECKING:
    from moment_to_action.metrics import MetricsCollector, Span
    from moment_to_action.models.vlm._base import LlamaVLModel

# (ModelID, display name)
_MODEL_CONFIGS: list[tuple[ModelID, str]] = [
    (ModelID.MOONDREAM2, "moondream2"),
    (ModelID.QWEN25_VL_3B_INSTRUCT, "qwen25_vl_3b"),
    (ModelID.QWEN3_VL_2B_INSTRUCT, "qwen3_vl_2b"),
    (ModelID.QWEN3_VL_4B_INSTRUCT, "qwen3_vl_4b"),
]

_N_CYCLES = 3
_MAX_TOKENS = 128
_N_FRAMES = 4  # synthetic frames per scene (duplicated still)

_YES_NO_LABELS = frozenset({"yes", "no"})

# Frame canvas dimensions.
_FRAME_W = 640
_FRAME_H = 480

# Background and label colors for synthetic frame rendering.
_BG_COLOR = (180, 180, 180)
_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (70, 130, 180),
    "dog": (139, 90, 43),
    "cat": (128, 0, 128),
    "bear": (101, 67, 33),
    "wolf": (100, 100, 100),
    "chair": (205, 133, 63),
    "desk": (139, 69, 19),
    "monitor": (47, 79, 79),
    "keyboard": (60, 60, 60),
    "laptop": (30, 30, 30),
    "cup": (220, 20, 60),
    "fork": (192, 192, 192),
    "sandwich": (255, 215, 0),
    "plate": (245, 245, 220),
    "dining table": (160, 82, 45),
    "hard hat": (255, 165, 0),
    "safety vest": (255, 255, 0),
    "glove": (0, 200, 0),
    "boot": (80, 80, 80),
}
_DEFAULT_COLOR = (150, 75, 0)


# ---------------------------------------------------------------------------
# Frame rendering / sourcing (raw BGR numpy arrays, matching RawFrameMessage)
# ---------------------------------------------------------------------------


def _render_frame(scene: Scene) -> Image.Image:
    """Render a synthetic RGB frame from a scene's bounding boxes.

    Draws each detection as a colored rectangle with a label on a gray canvas.
    The visual output encodes spatial arrangement and object identity without
    requiring real video footage.

    Args:
        scene: Scene whose detections to render.

    Returns:
        PIL Image (RGB, 640x480).
    """
    img = Image.new("RGB", (_FRAME_W, _FRAME_H), _BG_COLOR)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for det in scene.detections:
        color = _LABEL_COLORS.get(det.label, _DEFAULT_COLOR)
        b = det.bbox
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=3)
        label_text = f"{det.label} {det.confidence:.2f}"
        draw.text((b.x1 + 4, b.y1 + 4), label_text, fill=color, font=font)

    return img


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a BGR uint8 numpy array.

    Args:
        img: PIL Image (RGB).

    Returns:
        BGR uint8 array, as stored on ``RawFrameMessage``/``VideoClipMessage``.
    """
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _build_frames_synthetic(scene: Scene, n_frames: int) -> list[np.ndarray]:
    """Build synthetic BGR frames from scene bounding boxes.

    Renders one frame and duplicates it ``n_frames`` times so the VLM receives
    a consistent visual token sequence.

    Args:
        scene: Scene definition.
        n_frames: Number of frame copies to include.

    Returns:
        List of BGR uint8 frames, length ``n_frames``.
    """
    frame = _pil_to_bgr(_render_frame(scene))
    return [frame] * n_frames


def _sample_video_frames(video_path: Path, n_frames: int) -> list[np.ndarray]:
    """Sample ``n_frames`` uniformly from a video file as raw BGR frames.

    Args:
        video_path: Path to a video file readable by OpenCV.
        n_frames: Number of frames to sample.

    Returns:
        List of BGR uint8 frames, length up to ``n_frames``.

    Raises:
        RuntimeError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg = f"Cannot open video: {video_path}"
        raise RuntimeError(msg)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total / n_frames) for i in range(n_frames)] if total > 0 else []
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if ok:
            frames.append(bgr)
    cap.release()
    return frames


def _get_frames(
    scene: Scene, video_dir: Path | None, n_frames: int
) -> tuple[list[np.ndarray], bool]:
    """Return BGR frames for a scene, preferring real video over synthetic.

    Args:
        scene: Scene definition.
        video_dir: Optional directory to search for ``<scene_name>.mp4``.
        n_frames: Number of frames to sample or duplicate.

    Returns:
        Tuple of ``(frames, is_real)`` where ``is_real`` is True when real
        video was used.
    """
    if video_dir is not None:
        video_path = video_dir / f"{scene.name}.mp4"
        if video_path.exists():
            frames = _sample_video_frames(video_path, n_frames)
            if frames:
                return frames, True
    return _build_frames_synthetic(scene, n_frames), False


def _save_frames(scene: Scene, frames: list[np.ndarray], frames_dir: Path) -> None:
    """Save BGR frames to disk as JPEG files, via the same encoder the stage uses.

    Files are written as ``<frames_dir>/<scene_name>_<frame_idx>.jpg``.

    Args:
        scene: Scene whose frames to save (used for the filename prefix).
        frames: BGR uint8 frames.
        frames_dir: Directory to write frames into (must already exist).
    """
    for i, frame in enumerate(frames):
        img_bytes = base64.b64decode(bgr_to_b64(frame))
        dest = frames_dir / f"{scene.name}_{i:02d}.jpg"
        dest.write_bytes(img_bytes)


# ---------------------------------------------------------------------------
# Streaming benchmark
# ---------------------------------------------------------------------------


def _last_model_inference_span(metrics: MetricsCollector) -> Span | None:
    """Return the most recently recorded MODEL_INFERENCE span on *metrics*.

    Args:
        metrics: Collector to scan.

    Returns:
        The last :class:`~moment_to_action.metrics.Span` of type
        ``MODEL_INFERENCE``, or ``None`` if none have been recorded yet.
    """
    for span in reversed(metrics.spans):
        if span.type_ is SpanType.MODEL_INFERENCE:
            return span
    return None


def _run_scene(
    model: LlamaVLModel,
    model_name: str,
    scene: Scene,
    cycle: int,
    frames: list[np.ndarray],
    metrics: MetricsCollector,
) -> dict:
    """Drive one scene through ``Pipeline([VLMDescriptionStage, DecisionStage])`` and score it.

    Args:
        model: Loaded LlamaVLModel instance.
        model_name: Display name for the result row.
        scene: Scene to evaluate.
        cycle: Cycle index (1-based).
        frames: BGR uint8 frames for this scene.
        metrics: The same MetricsCollector *model* was constructed with.

    Returns:
        Result dict for this (scene, cycle).
    """
    is_yes_no = scene.expected_label.lower() in _YES_NO_LABELS
    grammar = YES_NO_GRAMMAR if is_yes_no else None
    clip_msg = VideoClipMessage(frames=frames, timestamp=0.0, question=scene.task)

    vlm_stage = VLMDescriptionStage(model, grammar=grammar, metrics=metrics)
    gen_messages = list(vlm_stage.process(iter([clip_msg])))
    gen_texts = [m for m in gen_messages if isinstance(m, GenerationMessage)]
    accumulated = gen_texts[-1].text if gen_texts else ""

    yn: str | None = None
    if is_yes_no:
        decision_stage = DecisionStage(metrics=metrics)
        decisions = [
            m for m in decision_stage.process(iter(gen_messages)) if isinstance(m, DecisionMessage)
        ]
        if decisions:
            yn = decisions[0].decision

    yn_correct: bool | None = None
    if yn is not None:
        yn_correct = yn == scene.expected_label.lower()

    span = _last_model_inference_span(metrics)
    meta = span.metadata if span is not None else {}
    inf_m = span.inference_metrics if span is not None else None

    return {
        "model": model_name,
        "app": scene.app,
        "scene": scene.name,
        "expected": scene.expected_label,
        "run_idx": cycle,
        "n_frames": len(frames),
        "response": accumulated,
        "response_chars": len(accumulated),
        "yn_correct": yn_correct,
        "recall": round(recall(accumulated, scene.recall_keywords), 4),
        "infer_ms": round(span.latency_ms, 3) if span is not None else None,
        "ttft_ms": meta.get("ttft_ms"),
        "ttfyd_ms": meta.get("ttfyd_ms"),
        "mean_itl_ms": meta.get("mean_itl_ms"),
        "std_itl_ms": meta.get("std_itl_ms"),
        "inference_metrics": inf_m.model_dump() if inf_m is not None else None,
    }


def _run_benchmark(
    model: LlamaVLModel,
    model_name: str,
    metrics: MetricsCollector,
    n_cycles: int,
    n_frames: int,
    video_dir: Path | None,
    progress: Progress,
    scene_task_id: TaskID,
) -> list[dict]:
    """Run all scenes x n_cycles through a loaded VLM model, return result rows.

    Args:
        model: Loaded LlamaVLModel instance.
        model_name: Human-readable name for output rows.
        metrics: The same MetricsCollector *model* was constructed with.
        n_cycles: Number of repetitions per scene.
        n_frames: Number of frames to pass per scene.
        video_dir: Optional directory with real video files.
        progress: Rich Progress instance for updating the scene sub-bar.
        scene_task_id: Task ID of the scene sub-progress bar.

    Returns:
        List of result dicts, one per (scene, cycle).
    """
    total_steps = len(SCENES) * n_cycles
    progress.reset(scene_task_id, total=total_steps)

    rows: list[dict] = []
    for cycle in range(1, n_cycles + 1):
        for scene in SCENES:
            progress.update(
                scene_task_id,
                description=f"  {scene.name} [{cycle}/{n_cycles}]",
            )
            frames, is_real = _get_frames(scene, video_dir, n_frames)
            try:
                row = _run_scene(model, model_name, scene, cycle, frames, metrics)
                row["real_video"] = is_real
            except Exception as exc:  # noqa: BLE001
                console.print(
                    f"  [yellow]{model_name} scene={scene.name} cycle={cycle}: {exc}[/yellow]"
                )
                progress.advance(scene_task_id)
                continue
            rows.append(row)
            progress.advance(scene_task_id)
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_summary(all_rows: list[dict]) -> None:
    """Print a rich summary table with per-model averages.

    Args:
        all_rows: All result rows from the benchmark.
    """
    groups: dict[str, list[dict]] = {}
    for row in all_rows:
        groups.setdefault(row["model"], []).append(row)

    table = Table(title="VLM Benchmark Summary", show_lines=True)
    table.add_column("Model", style="bold cyan")
    table.add_column("Load (ms)", justify="right")
    table.add_column("Unload (ms)", justify="right")
    table.add_column("Infer (ms)", justify="right")
    table.add_column("Response", justify="right")
    table.add_column("Recall", justify="right", style="bold green")
    table.add_column("YN Acc", justify="right", style="bold yellow")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("TTFYD (ms)", justify="right")
    table.add_column("Mean ITL (ms)", justify="right")

    for model_name, rows in sorted(groups.items()):
        avg_recall = float(np.mean([r["recall"] for r in rows]))
        yn_rows = [r for r in rows if r["yn_correct"] is not None]
        yn_acc = float(np.mean([r["yn_correct"] for r in yn_rows])) if yn_rows else float("nan")
        ttft_vals = [r["ttft_ms"] for r in rows if r["ttft_ms"] is not None]
        ttfyd_vals = [r["ttfyd_ms"] for r in rows if r["ttfyd_ms"] is not None]
        itl_vals = [r["mean_itl_ms"] for r in rows if r["mean_itl_ms"] is not None]
        load_vals = [r["load_ms"] for r in rows if r.get("load_ms") is not None]
        unload_vals = [r["unload_ms"] for r in rows if r.get("unload_ms") is not None]
        infer_vals = [r["infer_ms"] for r in rows if r["infer_ms"] is not None]
        resp_vals = [r["response_chars"] for r in rows]
        table.add_row(
            model_name,
            f"{float(np.mean(load_vals)):.0f}" if load_vals else "n/a",
            f"{float(np.mean(unload_vals)):.0f}" if unload_vals else "n/a",
            f"{float(np.mean(infer_vals)):.0f}" if infer_vals else "n/a",
            f"{float(np.mean(resp_vals)):.0f}" if resp_vals else "n/a",
            f"{avg_recall:.3f}",
            f"{yn_acc:.3f}" if yn_rows else "n/a",
            f"{float(np.mean(ttft_vals)):.1f}" if ttft_vals else "n/a",
            f"{float(np.mean(ttfyd_vals)):.1f}" if ttfyd_vals else "n/a",
            f"{float(np.mean(itl_vals)):.1f}" if itl_vals else "n/a",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark VLM models on application-specific classification scenes."
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=_N_CYCLES,
        help=f"Inference cycles per scene (default: {_N_CYCLES}).",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=_N_FRAMES,
        help=f"Frames per scene (default: {_N_FRAMES}).",
    )
    parser.add_argument(
        "--output",
        default="vlm_benchmark_results.json",
        help="Output JSON path (default: vlm_benchmark_results.json).",
    )
    parser.add_argument(
        "--server-path",
        default=None,
        help="Override llama_server_path from config.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override llama_server_port from config.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model display names to run (default: all). "
        "E.g. --models moondream2,qwen25_vl_3b",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_MAX_TOKENS,
        help=f"Maximum tokens per model response (default: {_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Directory with real video clips named <scene_name>.mp4. "
        "Falls back to synthetic frames when a clip is absent.",
    )
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Save frames for each scene to this directory as <scene>_<idx>.jpg. "
        "Directory is created if it does not exist.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference on CPU instead of GPU.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge results into existing output file instead of overwriting it.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Entry point for the VLM benchmark script."""
    args = _parse_args()
    n_cycles: int = args.n_cycles
    n_frames: int = args.n_frames
    max_tokens: int = args.max_tokens
    output_path = Path(args.output)
    compute_unit = ComputeUnit.CPU if args.cpu else ComputeUnit.GPU
    video_dir = Path(args.video_dir) if args.video_dir else None
    frames_dir = Path(args.frames_dir) if args.frames_dir else None

    ctx = build_context()
    server_path = Path(args.server_path) if args.server_path else ctx.config.llama_server_path
    port = args.port if args.port is not None else ctx.config.llama_server_port

    if server_path is None:
        console.print(
            "[red]llama_server_path not set. Use --server-path or set it in the M2A config.[/red]"
        )
        sys.exit(1)

    ctx.config = AppConfig(
        **{**ctx.config.model_dump(), "llama_server_path": server_path, "llama_server_port": port}
    )

    model_filter = set(args.models.split(",")) if args.models else None
    configs = [c for c in _MODEL_CONFIGS if model_filter is None or c[1] in model_filter]

    if not configs:
        console.print("[red]No models selected by filter. Exiting.[/red]")
        sys.exit(1)

    apps = sorted({s.app for s in SCENES})
    console.rule("[bold]M2A VLM Benchmark[/bold]")
    console.print(f"  apps   : {', '.join(apps)}")
    console.print(f"  scenes : {len(SCENES)}")
    console.print(f"  cycles : {n_cycles}")
    console.print(f"  frames : {n_frames} per scene")
    console.print(f"  tokens : {max_tokens}")
    console.print(f"  output : {output_path}")
    console.print(f"  models : {', '.join(c[1] for c in configs)}")
    console.print(f"  server : {server_path}:{port}")
    console.print(f"  device : {'CPU' if args.cpu else 'GPU'}")
    if video_dir:
        console.print(f"  videos : {video_dir}")
    else:
        console.print("  videos : synthetic (use --video-dir for real clips)")
    if frames_dir:
        console.print(f"  frames : saving to {frames_dir}")
    console.print()

    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for scene in SCENES:
            frames, _ = _get_frames(scene, video_dir, n_frames)
            _save_frames(scene, frames, frames_dir)
        console.print(f"[green]Frames saved to {frames_dir}[/green]\n")

    all_rows: list[dict] = []
    model_entries: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        model_task = progress.add_task("models", total=len(configs))
        scene_task = progress.add_task("  (waiting)", total=None)

        for model_id, model_name in configs:
            if model_id not in MODEL_REGISTRY:
                console.print(f"  [yellow]{model_name} not in registry, skipping.[/yellow]")
                progress.advance(model_task)
                continue

            progress.update(model_task, description=model_name)

            try:
                model = ctx.manager.get_model(
                    model_id,
                    system_prompt=BENCHMARK_SYSTEM,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [red]{model_name}: failed to get model — {exc}[/red]")
                progress.advance(model_task)
                continue

            rows: list[dict] = []
            with ctx.metrics.start_trace() as trace:
                try:
                    model.load(ctx.platform, compute_unit)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [red]{model_name}: failed to start — {exc}[/red]")
                    progress.advance(model_task)
                    continue

                rows = _run_benchmark(
                    model,  # type: ignore[arg-type]
                    model_name,
                    ctx.metrics,
                    n_cycles,
                    n_frames,
                    video_dir,
                    progress,
                    scene_task,
                )

                try:
                    model.unload()
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [yellow]{model_name}: unload error — {exc}[/yellow]")

            load_ms, unload_ms = extract_load_unload_ms([trace])
            for row in rows:
                row["load_ms"] = round(load_ms, 3)
                row["unload_ms"] = round(unload_ms, 3)
            model_entries.append(
                {
                    "model": model_name,
                    "load_ms": round(load_ms, 3),
                    "unload_ms": round(unload_ms, 3),
                    "trace": trace.json(),
                    "runs": rows,
                }
            )

            if rows:
                avg_recall = np.mean([r["recall"] for r in rows])
                console.print(
                    f"  [dim]{model_name}: {len(rows)} results — recall {avg_recall:.3f}[/dim]"
                )

            all_rows.extend(rows)
            progress.advance(model_task)

    if all_rows:
        write_results(
            model_entries,
            output_path,
            script="benchmark_vlms",
            key_fn=lambda e: e["model"],
            merge=args.merge,
        )
        console.print(f"\n[green]Results written to {output_path}[/green]")
    else:
        console.print("[yellow]No results produced.[/yellow]")

    console.print()
    _print_summary(all_rows)


if __name__ == "__main__":
    main()
