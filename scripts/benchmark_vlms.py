#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#     "moment-to-action",
#     "Pillow",
# ]
#
# [tool.uv.sources]
# moment-to-action = { path = "..", editable = true }
# ///
"""Benchmark VLM models on application-specific classification scenes with visual input.

Each scene maps to one of the five target applications (violence detection,
fall detection, animal threat, eating detection, PPE compliance). The VLM
receives video frames directly — rendered from the same scene bounding boxes
used in benchmark_llms.py — and answers the binary/multi-label question.

Two scenes per application: one positive case, one negative case.

By default, synthetic frames are generated from scene bounding box data using
PIL (colored rectangles with label text on a gray canvas). If ``--video-dir``
is supplied and a file ``<dir>/<scene_name>.mp4`` exists for a scene, real
frames are sampled from that video instead.

Accuracy metrics:
  - ``yn_correct``: bool — whether the model's first word was the correct yes/no.
  - ``recall``: float in [0, 1] — keyword recall for classification keywords.

Timing metrics (streaming-derived):
  - ``ttft_ms``: time from stream start to first token.
  - ``ttfyd_ms``: time from stream start to first yes/no decision.
  - ``mean_itl_ms``, ``std_itl_ms``: inter-token latency statistics.
  - ``inference_metrics``: llama.cpp-native timing fields from the stop chunk.

Usage:
    uv run python scripts/benchmark_vlms.py [--n-cycles 3] [--output results.json]

Requires llama_server_path to be set in the M2A config (or pass --server-path).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

from moment_to_action.config import AppConfig, load_config
from moment_to_action.hardware import ComputeUnit, Platform
from moment_to_action.hardware._loaded_models._llama import LlamaModel
from moment_to_action.metrics import MetricsCollector, SpanType
from moment_to_action.models import MODEL_REGISTRY, BaseModel, ModelID, ModelManager
from moment_to_action.models.image.detection._types import BoundingBox, Detection
from moment_to_action.paths import PathManager

if TYPE_CHECKING:
    from moment_to_action.hardware._metrics import LlamaCppInferenceMetrics
    from moment_to_action.metrics._types import MetricsReport

console = Console()

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

_BENCHMARK_SYSTEM = (
    "You are a scene analysis AI. Answer the user's question directly and concisely. "
    "Lead with your direct answer, then give one sentence of reasoning."
)

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
# Scene definition
# ---------------------------------------------------------------------------


def _bb(x1: int, y1: int, x2: int, y2: int) -> BoundingBox:
    """Construct a BoundingBox.

    Args:
        x1: Left edge.
        y1: Top edge.
        x2: Right edge.
        y2: Bottom edge.

    Returns:
        BoundingBox instance.
    """
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _det(label: str, conf: float, x1: int, y1: int, x2: int, y2: int) -> Detection:
    """Construct a Detection.

    Args:
        label: Class label.
        conf: Confidence score.
        x1: Left edge.
        y1: Top edge.
        x2: Right edge.
        y2: Bottom edge.

    Returns:
        Detection instance.
    """
    return Detection(label=label, confidence=conf, bbox=_bb(x1, y1, x2, y2))


@dataclass(frozen=True)
class Scene:
    """One benchmark scene backed by YOLO-realistic bounding box inputs.

    Attributes:
        name: Short identifier used in output.
        app: Target application name.
        task: The binary question posed to the VLM.
        detections: YOLO detections used to render the synthetic frame.
        expected_label: Correct answer token (e.g. "YES", "NO", "COMPLIANT").
        recall_keywords: Words expected from a correct answer.
    """

    name: str
    app: str
    task: str
    detections: list[Detection]
    expected_label: str
    recall_keywords: list[str] = field(default_factory=list)


_SCENES: list[Scene] = [
    # --- Violence Detection -------------------------------------------------
    Scene(
        name="violence_fight",
        app="violence_detection",
        task=(
            "Is a violent incident occurring in this scene? "
            "Answer YES or NO, then one sentence of reasoning."
        ),
        detections=[
            _det("person", 0.95, 80, 40, 360, 480),
            _det("person", 0.92, 200, 30, 500, 480),
        ],
        expected_label="YES",
        recall_keywords=["yes", "fight", "violen", "aggress", "altercation", "physical"],
    ),
    Scene(
        name="violence_calm",
        app="violence_detection",
        task=(
            "Is a violent incident occurring in this scene? "
            "Answer YES or NO, then one sentence of reasoning."
        ),
        detections=[
            _det("person", 0.93, 10, 50, 200, 480),
            _det("person", 0.90, 440, 50, 630, 480),
        ],
        expected_label="NO",
        recall_keywords=["no", "calm", "peaceful", "safe", "non-violent", "normal"],
    ),
    # --- Fall Detection -----------------------------------------------------
    Scene(
        name="fall_detected",
        app="fall_detection",
        task="Has a person fallen in this scene? Answer YES or NO, then one sentence of reasoning.",
        detections=[
            _det("person", 0.91, 50, 390, 520, 470),
            _det("chair", 0.74, 300, 200, 500, 400),
        ],
        expected_label="YES",
        recall_keywords=["yes", "fall", "fallen", "ground", "floor", "horizontal", "lying"],
    ),
    Scene(
        name="fall_standing",
        app="fall_detection",
        task="Has a person fallen in this scene? Answer YES or NO, then one sentence of reasoning.",
        detections=[
            _det("person", 0.95, 220, 40, 400, 480),
            _det("desk", 0.81, 400, 200, 640, 480),
            _det("monitor", 0.78, 460, 60, 620, 260),
        ],
        expected_label="NO",
        recall_keywords=["no", "standing", "upright", "vertical", "normal", "not fallen"],
    ),
    # --- Animal Threat ------------------------------------------------------
    Scene(
        name="animal_threat",
        app="animal_threat_detection",
        task=(
            "Is an animal posing an immediate threat to a person in this scene? "
            "Answer YES or NO, then one sentence of reasoning."
        ),
        detections=[
            _det("person", 0.93, 150, 80, 430, 480),
            _det("dog", 0.88, 350, 180, 620, 480),
        ],
        expected_label="YES",
        recall_keywords=["yes", "threat", "danger", "aggress", "attack", "immediate"],
    ),
    Scene(
        name="animal_safe",
        app="animal_threat_detection",
        task=(
            "Is an animal posing an immediate threat to a person in this scene? "
            "Answer YES or NO, then one sentence of reasoning."
        ),
        detections=[
            _det("person", 0.94, 80, 50, 380, 480),
            _det("dog", 0.76, 530, 320, 610, 400),
        ],
        expected_label="NO",
        recall_keywords=["no", "safe", "distant", "no threat", "away", "not immediate"],
    ),
    # --- Eating Detection ---------------------------------------------------
    Scene(
        name="eating_yes",
        app="eating_detection",
        task=(
            "Egocentric view from a wearable camera. "
            "Is the wearer currently eating or drinking? "
            "Answer YES or NO, then one sentence of reasoning."
        ),
        detections=[
            _det("fork", 0.89, 240, 300, 400, 440),
            _det("sandwich", 0.84, 140, 270, 450, 460),
            _det("plate", 0.91, 70, 260, 580, 470),
            _det("dining table", 0.72, 0, 410, 640, 480),
        ],
        expected_label="YES",
        recall_keywords=["yes", "eating", "meal", "consuming", "food", "fork"],
    ),
    Scene(
        name="eating_no",
        app="eating_detection",
        task=(
            "Egocentric view from a wearable camera. "
            "Is the wearer currently eating or drinking? "
            "Answer YES or NO, then one sentence of reasoning."
        ),
        detections=[
            _det("keyboard", 0.93, 90, 360, 550, 470),
            _det("laptop", 0.88, 140, 200, 500, 400),
            _det("monitor", 0.85, 40, 40, 600, 300),
            _det("cup", 0.65, 575, 360, 635, 440),
        ],
        expected_label="NO",
        recall_keywords=["no", "working", "typing", "not eating", "computer", "keyboard"],
    ),
    # --- PPE Compliance -----------------------------------------------------
    Scene(
        name="ppe_compliant",
        app="ppe_compliance",
        task=(
            "Is the construction worker wearing all required PPE "
            "(hard hat, safety vest, gloves, boots)? "
            "Answer COMPLIANT or NON-COMPLIANT, then list present and missing items."
        ),
        detections=[
            _det("person", 0.96, 120, 40, 520, 480),
            _det("hard hat", 0.91, 230, 40, 420, 140),
            _det("safety vest", 0.88, 140, 150, 500, 340),
            _det("glove", 0.79, 120, 310, 230, 420),
            _det("glove", 0.77, 410, 310, 520, 420),
            _det("boot", 0.83, 160, 410, 290, 480),
            _det("boot", 0.80, 350, 410, 480, 480),
        ],
        expected_label="COMPLIANT",
        recall_keywords=["compliant", "hat", "vest", "glove", "boot", "all", "present"],
    ),
    Scene(
        name="ppe_violation",
        app="ppe_compliance",
        task=(
            "Is the construction worker wearing all required PPE "
            "(hard hat, safety vest, gloves, boots)? "
            "Answer COMPLIANT or NON-COMPLIANT, then list present and missing items."
        ),
        detections=[
            _det("person", 0.95, 120, 40, 520, 480),
            _det("safety vest", 0.90, 140, 150, 500, 340),
            _det("boot", 0.84, 160, 410, 290, 480),
            _det("boot", 0.82, 350, 410, 480, 480),
        ],
        expected_label="NON-COMPLIANT",
        recall_keywords=["non-compliant", "missing", "hat", "glove", "absent", "violation"],
    ),
]


# ---------------------------------------------------------------------------
# Synthetic frame builder
# ---------------------------------------------------------------------------


def _render_frame(scene: Scene) -> Image.Image:
    """Render a single synthetic PIL frame from scene bounding boxes.

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


def _pil_to_base64(img: Image.Image) -> str:
    """Encode a PIL image as a base64 JPEG string.

    Args:
        img: PIL Image to encode.

    Returns:
        Base64-encoded JPEG bytes as a UTF-8 string (no ``data:`` prefix).
    """
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _build_frames_synthetic(scene: Scene, n_frames: int) -> list[str]:
    """Build synthetic base64 JPEG frames from scene bounding boxes.

    Renders one frame and duplicates it ``n_frames`` times so the VLM receives
    a consistent visual token sequence.

    Args:
        scene: Scene definition.
        n_frames: Number of frame copies to include.

    Returns:
        List of base64-encoded JPEG strings, length ``n_frames``.
    """
    frame = _render_frame(scene)
    b64 = _pil_to_base64(frame)
    return [b64] * n_frames


def _sample_video_frames(video_path: Path, n_frames: int) -> list[str]:
    """Sample ``n_frames`` uniformly from a video file and return as base64 JPEGs.

    Args:
        video_path: Path to a video file readable by OpenCV.
        n_frames: Number of frames to sample.

    Returns:
        List of base64-encoded JPEG strings, length up to ``n_frames``.

    Raises:
        RuntimeError: If the video cannot be opened.
    """
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg = f"Cannot open video: {video_path}"
        raise RuntimeError(msg)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total / n_frames) for i in range(n_frames)] if total > 0 else []
    frames: list[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        frames.append(_pil_to_base64(img))
    cap.release()
    return frames


def _get_frames(scene: Scene, video_dir: Path | None, n_frames: int) -> tuple[list[str], bool]:
    """Return base64 frames for a scene, preferring real video over synthetic.

    Args:
        scene: Scene definition.
        video_dir: Optional directory to search for ``<scene_name>.mp4``.
        n_frames: Number of frames to sample or duplicate.

    Returns:
        Tuple of ``(b64_frames, is_real)`` where ``is_real`` is True when real
        video was used.
    """
    if video_dir is not None:
        video_path = video_dir / f"{scene.name}.mp4"
        if video_path.exists():
            frames = _sample_video_frames(video_path, n_frames)
            if frames:
                return frames, True
    return _build_frames_synthetic(scene, n_frames), False


def _save_frames(scene: Scene, b64_frames: list[str], frames_dir: Path) -> None:
    """Save base64-encoded frames to disk as JPEG files.

    Files are written as ``<frames_dir>/<scene_name>_<frame_idx>.jpg``.

    Args:
        scene: Scene whose frames to save (used for the filename prefix).
        b64_frames: List of base64-encoded JPEG strings.
        frames_dir: Directory to write frames into (must already exist).
    """
    for i, b64 in enumerate(b64_frames):
        img_bytes = base64.b64decode(b64)
        dest = frames_dir / f"{scene.name}_{i:02d}.jpg"
        dest.write_bytes(img_bytes)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _recall(response: str, keywords: list[str]) -> float:
    """Compute keyword recall: fraction of expected classification keywords found.

    Args:
        response: Text generated by the model.
        keywords: Words expected from a correct answer.

    Returns:
        Fraction in [0, 1] of keywords found (case-insensitive).
    """
    if not keywords:
        return 1.0
    resp_lower = response.lower()
    found = sum(1 for kw in keywords if kw.lower() in resp_lower)
    return found / len(keywords)


def _detect_yn(text: str) -> str | None:
    """Return "yes" or "no" if the first word of ``text`` is a yes/no answer, else ``None``.

    Args:
        text: Accumulated model response so far.

    Returns:
        "yes", "no", or ``None`` if not yet decidable.
    """
    words = text.strip().lower().split()
    if not words:
        return None
    first = words[0].rstrip(".,!?;:")
    return first if first in {"yes", "no"} else None


def _extract_load_unload_ms(report: MetricsReport) -> tuple[float, float]:
    """Extract load and unload latencies from a completed metrics report.

    Args:
        report: A completed MetricsReport whose trace spans are accessible.

    Returns:
        Tuple of (load_ms, unload_ms).  Returns 0.0 for any span not found.
    """
    load_ms = 0.0
    unload_ms = 0.0
    for trace in report.traces:
        for span in trace.spans:
            if span.type_ == SpanType.MODEL_LOAD:
                load_ms = span.latency_ms
            elif span.type_ == SpanType.MODEL_UNLOAD:
                unload_ms = span.latency_ms
    return load_ms, unload_ms


# ---------------------------------------------------------------------------
# Streaming benchmark
# ---------------------------------------------------------------------------


def _run_scene(
    model: BaseModel,
    model_name: str,
    scene: Scene,
    cycle: int,
    b64_frames: list[str],
    metrics: MetricsCollector,
) -> dict:
    """Stream one scene through the VLM model and collect all metrics.

    Wraps the streaming loop in a MODEL_INFERENCE span.  Tracks TTFT, TTFYD,
    and inter-token latencies manually via perf_counter, then stores them as
    span metadata via ``metrics.set_meta``.

    Args:
        model: Loaded LlamaVLModel instance.
        model_name: Display name for the result.
        scene: Scene to evaluate.
        cycle: Cycle index (1-based).
        b64_frames: Base64-encoded JPEG frames for this scene.
        metrics: Active MetricsCollector (a trace must be open).

    Returns:
        Result dict for this (scene, cycle).

    Raises:
        RuntimeError: If ``model._loaded_model`` is not a LlamaModel.
    """
    loaded_model = getattr(model, "_loaded_model", None)
    if not isinstance(loaded_model, LlamaModel):
        msg = f"{model_name}: _loaded_model is not a LlamaModel"
        raise TypeError(msg)

    prepared = model.prepare((scene.task, b64_frames), metrics=metrics)

    t0 = time.perf_counter_ns()
    ttft_ms: float | None = None
    ttfyd_ms: float | None = None
    token_times: list[int] = []
    accumulated = ""
    inf_m: LlamaCppInferenceMetrics | None = None

    with metrics.start_span(SpanType.MODEL_INFERENCE, f"{model_name}.stream") as span:
        for token in loaded_model.stream(prepared):
            now = time.perf_counter_ns()
            token_times.append(now)
            accumulated += token
            if ttft_ms is None:
                ttft_ms = (now - t0) / 1e6
            if ttfyd_ms is None and _detect_yn(accumulated) is not None:
                ttfyd_ms = (now - t0) / 1e6

        t_end = time.perf_counter_ns()
        itl_ms = [(token_times[i] - token_times[i - 1]) / 1e6 for i in range(1, len(token_times))]
        mean_itl = float(np.mean(itl_ms)) if itl_ms else 0.0
        std_itl = float(np.std(itl_ms)) if itl_ms else 0.0
        infer_ms = (t_end - t0) / 1e6

        inf_m = loaded_model.last_inference_metrics
        if inf_m is not None:
            span.inference_metrics = inf_m

        metrics.set_meta("ttft_ms", ttft_ms)
        metrics.set_meta("ttfyd_ms", ttfyd_ms)
        metrics.set_meta("mean_itl_ms", round(mean_itl, 3))
        metrics.set_meta("std_itl_ms", round(std_itl, 3))

    yn = _detect_yn(accumulated)
    yn_correct: bool | None = None
    if yn is not None:
        yn_correct = yn == scene.expected_label.lower()

    return {
        "model": model_name,
        "app": scene.app,
        "scene": scene.name,
        "expected": scene.expected_label,
        "run_idx": cycle,
        "n_frames": len(b64_frames),
        "response": accumulated,
        "response_chars": len(accumulated),
        "yn_correct": yn_correct,
        "recall": round(_recall(accumulated, scene.recall_keywords), 4),
        "infer_ms": round(infer_ms, 3),
        "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
        "ttfyd_ms": round(ttfyd_ms, 3) if ttfyd_ms is not None else None,
        "mean_itl_ms": round(mean_itl, 3),
        "std_itl_ms": round(std_itl, 3),
        "inference_metrics": inf_m.model_dump() if inf_m is not None else None,
    }


def _run_benchmark(
    model: BaseModel,
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
        metrics: Active MetricsCollector (a trace must be open).
        n_cycles: Number of repetitions per scene.
        n_frames: Number of frames to pass per scene.
        video_dir: Optional directory with real video files.
        progress: Rich Progress instance for updating the scene sub-bar.
        scene_task_id: Task ID of the scene sub-progress bar.

    Returns:
        List of result dicts, one per (scene, cycle).
    """
    total_steps = len(_SCENES) * n_cycles
    progress.reset(scene_task_id, total=total_steps)

    rows: list[dict] = []
    for cycle in range(1, n_cycles + 1):
        for scene in _SCENES:
            progress.update(
                scene_task_id,
                description=f"  {scene.name} [{cycle}/{n_cycles}]",
            )
            b64_frames, is_real = _get_frames(scene, video_dir, n_frames)
            try:
                row = _run_scene(model, model_name, scene, cycle, b64_frames, metrics)
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
        itl_vals = [r["mean_itl_ms"] for r in rows]
        load_vals = [r["load_ms"] for r in rows if r.get("load_ms") is not None]
        unload_vals = [r["unload_ms"] for r in rows if r.get("unload_ms") is not None]
        infer_vals = [r["infer_ms"] for r in rows]
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


def _write_json(model_entries: list[dict], output_path: Path) -> None:
    """Write benchmark results to a JSON file.

    Each entry in ``model_entries`` contains a single model's ``metrics_report``
    (once, not duplicated per row) plus a ``runs`` list of per-scene result dicts.

    Args:
        model_entries: Per-model result dicts, each containing ``metrics_report`` and ``runs``.
        output_path: Destination JSON path.
    """
    output = {
        "script": "benchmark_vlms",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "models": model_entries,
    }
    output_path.write_text(json.dumps(output, indent=2))


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
    return parser.parse_args()


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Entry point for the VLM benchmark script."""
    global _N_CYCLES, _MAX_TOKENS, _N_FRAMES  # noqa: PLW0603

    args = _parse_args()
    _N_CYCLES = args.n_cycles
    _MAX_TOKENS = args.max_tokens
    _N_FRAMES = args.n_frames
    output_path = Path(args.output)
    video_dir = Path(args.video_dir) if args.video_dir else None
    frames_dir = Path(args.frames_dir) if args.frames_dir else None

    path_manager = PathManager()
    config = load_config(path_manager.app_config_file)
    server_path = Path(args.server_path) if args.server_path else config.llama_server_path
    port = args.port if args.port is not None else config.llama_server_port

    if server_path is None:
        console.print(
            "[red]llama_server_path not set. Use --server-path or set it in the M2A config.[/red]"
        )
        sys.exit(1)

    config = AppConfig(
        **{**config.model_dump(), "llama_server_path": server_path, "llama_server_port": port}
    )

    model_filter = set(args.models.split(",")) if args.models else None
    configs = [c for c in _MODEL_CONFIGS if model_filter is None or c[1] in model_filter]

    if not configs:
        console.print("[red]No models selected by filter. Exiting.[/red]")
        sys.exit(1)

    apps = sorted({s.app for s in _SCENES})
    console.rule("[bold]M2A VLM Benchmark[/bold]")
    console.print(f"  apps   : {', '.join(apps)}")
    console.print(f"  scenes : {len(_SCENES)}")
    console.print(f"  cycles : {_N_CYCLES}")
    console.print(f"  frames : {_N_FRAMES} per scene")
    console.print(f"  tokens : {_MAX_TOKENS}")
    console.print(f"  output : {output_path}")
    console.print(f"  models : {', '.join(c[1] for c in configs)}")
    console.print(f"  server : {server_path}:{port}")
    if video_dir:
        console.print(f"  videos : {video_dir}")
    else:
        console.print("  videos : synthetic (use --video-dir for real clips)")
    if frames_dir:
        console.print(f"  frames : saving to {frames_dir}")
    console.print()

    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for scene in _SCENES:
            b64_frames, _ = _get_frames(scene, video_dir, _N_FRAMES)
            _save_frames(scene, b64_frames, frames_dir)
        console.print(f"[green]Frames saved to {frames_dir}[/green]\n")

    manager = ModelManager(path_manager)
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

            platform = Platform(config)
            metrics = MetricsCollector(platform)
            try:
                model = manager.get_model(
                    model_id,
                    system_prompt=_BENCHMARK_SYSTEM,
                    max_tokens=_MAX_TOKENS,
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [red]{model_name}: failed to get model — {exc}[/red]")
                progress.advance(model_task)
                continue

            rows: list[dict] = []
            with metrics.start_trace():
                try:
                    model.load(platform, ComputeUnit.GPU, metrics=metrics)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [red]{model_name}: failed to start — {exc}[/red]")
                    progress.advance(model_task)
                    continue

                rows = _run_benchmark(
                    model,
                    model_name,
                    metrics,
                    _N_CYCLES,
                    _N_FRAMES,
                    video_dir,
                    progress,
                    scene_task,
                )

                try:
                    model.unload(metrics=metrics)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [yellow]{model_name}: unload error — {exc}[/yellow]")

            report = metrics.report()
            load_ms, unload_ms = _extract_load_unload_ms(report)
            for row in rows:
                row["load_ms"] = round(load_ms, 3)
                row["unload_ms"] = round(unload_ms, 3)
            model_entries.append(
                {
                    "model": model_name,
                    "load_ms": round(load_ms, 3),
                    "unload_ms": round(unload_ms, 3),
                    "metrics_report": report.json(),
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
        _write_json(model_entries, output_path)
        console.print(f"\n[green]Results written to {output_path}[/green]")
    else:
        console.print("[yellow]No results produced.[/yellow]")

    console.print()
    _print_summary(all_rows)


if __name__ == "__main__":
    main()
