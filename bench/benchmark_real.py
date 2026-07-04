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
"""Benchmark all VLMs and LLMs against real annotated video clips.

Reads ``bench/data/annotations.json`` (or ``--data-dir``/annotations.json) which
maps three applications — violence_detection, eating, animals — to a list of
video clips.  Each clip has a ground-truth ``label`` ("positive" / "negative"),
``start_s`` / ``end_s`` timestamps that window the ROI, and a relative path to the
video file inside ``<data-dir>/videos/``.

Both pipelines are driven through :class:`~moment_to_action.Moment2Action` — one
pipeline per model (VLM) or per (detector, LLM) pair, loaded once and reused
across every clip and cycle. The per-clip ``question`` rides on the message
(``DetectionMessage.question`` / ``VideoClipMessage.question``) rather than being
fixed at stage construction, so one loaded model serves every application's
question without reloading.

VLM pipeline — one real, chained ``Moment2Action`` pipeline per model:
  1. Extract frames at 1 FPS from the annotated ROI window, resize to <=480px tall.
  2. ``app.new_pipeline(...).add_stage(VLMDescriptionStage, ..., grammar=YES_NO_GRAMMAR)
     .add_stage(DecisionStage).build()``, loaded once, run once per
     ``VideoClipMessage(frames, question=clip.question)``.
  3. Score the streamed response and collect full timing + accuracy metrics.

LLM pipeline — one ``Moment2Action`` pipeline per (detector, LLM) pair holding both
models loaded together; the per-clip frame count varies, so the detection ->
aggregation -> LLM hop can't be expressed as a single chained ``run()`` (aggregation
window size isn't known until each clip's frames are extracted). The detection and
LLM/decision stages are pulled off the loaded pipeline and driven directly, wrapped
in ``handle.trace()`` so their spans still land on this pipeline's own metrics:
  1. Extract frames at 1 FPS from the annotated ROI window.
  2. Run the pipeline's ``ImageDetectionStage`` on each frame.
  3. Aggregate detections across all frames via a freshly built
     ``DetectionAggregationStage(window=len(frames))`` (keep highest-confidence
     instance per label) — the aggregated ``DetectionMessage`` carries ``clip.question``.
  4. Run the pipeline's ``LLMStage``/``DecisionStage`` on the aggregated message.
  5. Score the streamed response and collect full timing + accuracy metrics.

Model lists at the top of this file — comment out any entry to skip it.

Usage:
    uv run python bench/benchmark_real.py [options]

Requires ``llama_server_path`` to be set in the M2A config (or pass ``--server-path``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from _common import console
from rich.console import Console
from rich.logging import RichHandler
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

from moment_to_action import Moment2Action
from moment_to_action.benchmarking import extract_load_unload_ms, recall
from moment_to_action.config import AppConfig, load_config
from moment_to_action.hardware import ComputeUnit
from moment_to_action.messages import DecisionMessage, DetectionMessage, RawFrameMessage
from moment_to_action.messages.video import VideoClipMessage
from moment_to_action.metrics import SpanType
from moment_to_action.models import MODEL_REGISTRY, ModelID
from moment_to_action.paths import PathManager
from moment_to_action.prompting import BENCHMARK_SYSTEM, CHATML, PHI3, YES_NO_GRAMMAR
from moment_to_action.stages.image import DetectionAggregationStage, ImageDetectionStage
from moment_to_action.stages.llm import DecisionStage, LLMStage
from moment_to_action.stages.vlm import VLMDescriptionStage

if TYPE_CHECKING:
    from moment_to_action.app import PipelineHandle
    from moment_to_action.metrics import Span, Trace
    from moment_to_action.models.image.detection._types import Detection

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=Console(stderr=True))],
)

_YES_NO_LABELS = frozenset({"yes", "no"})


# ---------------------------------------------------------------------------
# Model lists — comment out any entry to skip it
# ---------------------------------------------------------------------------

# (ModelID, display name)
_VLM_CONFIGS: list[tuple[ModelID, str]] = [
    (ModelID.MOONDREAM2, "moondream2"),
    (ModelID.SMOLVLM2_256M, "smolvlm2_256m"),
    (ModelID.SMOLVLM2_500M, "smolvlm2_500m"),
    (ModelID.SMOLVLM2_2_2B, "smolvlm2_2_2b"),
    (ModelID.QWEN25_VL_3B_INSTRUCT, "qwen25_vl_3b"),
    (ModelID.QWEN3_VL_2B_INSTRUCT, "qwen3_vl_2b"),
    # (ModelID.QWEN3_VL_4B_INSTRUCT, "qwen3_vl_4b"),
    (ModelID.INTERNVL3_1B_INSTRUCT, "internvl3_1b"),
    (ModelID.MINISTRAL_3_3B_REASONING, "ministral_3_3b"),
]

# (ModelID, display name, prompt template | None)
_LLM_CONFIGS: list[tuple[ModelID, str, str | None]] = [
    (ModelID.QWEN3_0_6B, "qwen3_0_6b", CHATML),
    (ModelID.QWEN3_1_7B, "qwen3_1_7b", CHATML),
    (ModelID.GEMMA3_270M_IT, "gemma3_270m", CHATML),
    (ModelID.GEMMA3_1B_IT, "gemma3_1b", CHATML),
    (ModelID.QWEN2_1_5B_INSTRUCT, "qwen2_1_5b", CHATML),
    (ModelID.QWEN3_4B, "qwen3_4b", CHATML),
    (ModelID.PHI35_MINI_INSTRUCT, "phi35_mini", PHI3),
]

# Detectors used for the LLM pipeline — comment out any to skip.
# Tuple: (ModelID, display_name, ComputeUnit, variant_key)
_LLM_DETECTORS: list[tuple[ModelID, str, ComputeUnit, str]] = [
    # (ModelID.YOLO_V8, "yolo_v8", ComputeUnit.NPU, "qcs6490"),
    (ModelID.DETECTRON2, "detectron2", ComputeUnit.NPU, "qcs6490_w8a16"),
]

_N_CYCLES = 3
_MAX_TOKENS = 128

_MAX_FRAME_HEIGHT = 480


# ---------------------------------------------------------------------------
# Annotation schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Clip:
    """One annotated video clip from annotations.json.

    Attributes:
        id: Unique clip identifier.
        file: Relative path from ``<data_dir>/videos/`` to the video file.
        label: Ground-truth label — "positive" or "negative".
        start_s: Start of the ROI window in seconds.
        end_s: End of the ROI window in seconds; ``None`` means end of file.
        description: Human-readable description of the clip content.
        application: Parent application name (injected when loading).
        question: Application question to pose to the model (injected when loading).
        positive_keywords: Recall keywords for a correct positive answer.
        negative_keywords: Recall keywords for a correct negative answer.
    """

    id: str
    file: str
    label: str
    start_s: float
    end_s: float | None
    description: str
    application: str
    question: str
    positive_keywords: list[str] = field(default_factory=list)
    negative_keywords: list[str] = field(default_factory=list)

    @property
    def expected(self) -> str:
        """Return the expected YES/NO token for this clip.

        Returns:
            "YES" for positive clips, "NO" for negative clips.
        """
        return "YES" if self.label == "positive" else "NO"

    @property
    def recall_keywords(self) -> list[str]:
        """Return the recall keywords matching the expected answer.

        Returns:
            positive_keywords for positive clips, negative_keywords for negative clips.
        """
        return self.positive_keywords if self.label == "positive" else self.negative_keywords


def _load_clips(data_dir: Path) -> list[Clip]:
    """Load and validate clips from annotations.json.

    Args:
        data_dir: Root data directory containing ``annotations.json``.

    Returns:
        Flat list of Clip objects across all applications.

    Raises:
        FileNotFoundError: If annotations.json does not exist.
    """
    ann_path = data_dir / "annotations.json"
    if not ann_path.exists():
        msg = f"annotations.json not found at {ann_path}"
        raise FileNotFoundError(msg)
    raw = json.loads(ann_path.read_text())
    clips: list[Clip] = []
    for app_name, app_data in raw.get("applications", {}).items():
        question = app_data.get("question", "")
        pos_kw = app_data.get("positive_keywords", [])
        neg_kw = app_data.get("negative_keywords", [])
        clips.extend(
            Clip(
                id=c["id"],
                file=c["file"],
                label=c["label"],
                start_s=float(c.get("start_s", 0.0)),
                end_s=float(c["end_s"]) if c.get("end_s") is not None else None,
                description=c.get("description", ""),
                application=app_name,
                question=question,
                positive_keywords=pos_kw,
                negative_keywords=neg_kw,
            )
            for c in app_data.get("clips", [])
        )
    return clips


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def _extract_frames_1fps(
    video_path: Path,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> list[np.ndarray]:
    """Extract one frame per second from a video ROI window using OpenCV.

    Args:
        video_path: Path to a video file (H.264 mp4 recommended).
        start_s: Start of the ROI window in seconds.
        end_s: End of the ROI window in seconds; ``None`` means end of file.

    Returns:
        List of BGR uint8 frames sampled at 1 FPS within [start_s, end_s].

    Raises:
        RuntimeError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg = f"Cannot open video: {video_path}"
        raise RuntimeError(msg)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps

    start_frame = int(start_s * fps)
    end_frame = int((end_s if end_s is not None else duration_s) * fps)
    end_frame = min(end_frame, total_frames)

    step = max(1, round(fps))
    frames: list[np.ndarray] = []
    for i in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def _resize_480p(frame: np.ndarray) -> np.ndarray:
    """Resize a frame to at most 480 px tall, preserving aspect ratio.

    Args:
        frame: BGR uint8 image array.

    Returns:
        Resized frame, or original if already <= 480 px tall.
    """
    h, w = frame.shape[:2]
    if h <= _MAX_FRAME_HEIGHT:
        return frame
    scale = _MAX_FRAME_HEIGHT / h
    return cv2.resize(frame, (int(w * scale), _MAX_FRAME_HEIGHT), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Streaming benchmark
# ---------------------------------------------------------------------------


def _last_model_inference_span(trace: Trace) -> Span | None:
    """Return the most recently recorded MODEL_INFERENCE span on *trace*.

    Args:
        trace: Trace to scan (typically one clip's evaluation).

    Returns:
        The last :class:`~moment_to_action.metrics.Span` of type
        ``MODEL_INFERENCE``, or ``None`` if none were recorded.
    """
    for span in reversed(trace.spans):
        if span.type_ is SpanType.MODEL_INFERENCE:
            return span
    return None


def _score_clip_response(
    accumulated: str, clip: Clip, trace: Trace, *, is_yes_no: bool, yn: str | None
) -> dict:
    """Build the common scoring fields shared by the VLM and LLM clip runners.

    Args:
        accumulated: Full generated response text.
        clip: Clip being evaluated.
        trace: This clip's evaluation trace.
        is_yes_no: Whether a decision was attempted for this clip.
        yn: Extracted decision ("yes"/"no"), or ``None`` if not decided.

    Returns:
        Dict of scoring/timing fields common to VLM and LLM result rows.
    """
    yn_correct: bool | None = None
    if is_yes_no and yn is not None:
        yn_correct = yn == clip.expected.lower()

    span = _last_model_inference_span(trace)
    meta = span.metadata if span is not None else {}
    inf_m = span.inference_metrics if span is not None else None

    return {
        "response": accumulated,
        "response_chars": len(accumulated),
        "yn_correct": yn_correct,
        "recall": round(recall(accumulated, clip.recall_keywords), 4),
        "infer_ms": round(span.latency_ms, 3) if span is not None else None,
        "ttft_ms": meta.get("ttft_ms"),
        "ttfyd_ms": meta.get("ttfyd_ms"),
        "mean_itl_ms": meta.get("mean_itl_ms"),
        "std_itl_ms": meta.get("std_itl_ms"),
        "inference_metrics": inf_m.model_dump() if inf_m is not None else None,
    }


def _run_vlm_clip(
    handle: PipelineHandle,
    model_name: str,
    clip: Clip,
    cycle: int,
    frames: list[np.ndarray],
) -> dict:
    """Drive one clip through the loaded VLM+Decision pipeline and score it.

    Args:
        handle: Loaded pipeline: ``[VLMDescriptionStage, DecisionStage]``.
        model_name: Display name for the result row.
        clip: Clip being evaluated.
        cycle: Cycle index (1-based).
        frames: BGR uint8 frames for this clip's ROI (already resized to <=480px tall).

    Returns:
        Result dict for this (clip, cycle).
    """
    is_yes_no = clip.expected.lower() in _YES_NO_LABELS
    vlm_stage, decision_stage = handle.stages
    clip_msg = VideoClipMessage(frames=frames, timestamp=time.time(), question=clip.question)

    with handle.trace() as trace:
        gen_messages = list(vlm_stage.process(iter([clip_msg])))
        accumulated = gen_messages[-1].text if gen_messages else ""  # type: ignore[union-attr]

        yn: str | None = None
        if is_yes_no:
            decisions = [
                m
                for m in decision_stage.process(iter(gen_messages))
                if isinstance(m, DecisionMessage)
            ]
            if decisions:
                yn = decisions[0].decision

    frame_h, frame_w = frames[0].shape[:2]
    return {
        "model": model_name,
        "application": clip.application,
        "clip_id": clip.id,
        "video_file": clip.file,
        "label": clip.label,
        "start_s": clip.start_s,
        "end_s": clip.end_s,
        "expected": clip.expected,
        "run_idx": cycle,
        "n_frames": len(frames),
        "frame_w": frame_w,
        "frame_h": frame_h,
        **_score_clip_response(accumulated, clip, trace, is_yes_no=is_yes_no, yn=yn),
    }


def _run_llm_clip(
    handle: PipelineHandle,
    model_name: str,
    clip: Clip,
    cycle: int,
    detections: list[Detection],
    n_frames: int,
) -> dict:
    """Drive one clip through the loaded LLM+Decision stages and score it.

    The prompt is built from aggregated detections; no images are passed. Pulls
    ``LLMStage``/``DecisionStage`` directly off *handle* (see module docstring for
    why this pipeline can't be driven with one chained ``run()``), wrapped in
    ``handle.trace()`` so spans land on this pipeline's own metrics.

    Args:
        handle: Loaded pipeline: ``[ImageDetectionStage, LLMStage, DecisionStage]``.
        model_name: Display name for the result row.
        clip: Clip being evaluated.
        cycle: Cycle index (1-based).
        detections: Aggregated representative detections from the clip's ROI.
        n_frames: Number of frames that were extracted (for metadata).

    Returns:
        Result dict for this (clip, cycle).
    """
    is_yes_no = clip.expected.lower() in _YES_NO_LABELS
    _, llm_stage, decision_stage = handle.stages
    detection_msg = DetectionMessage(
        timestamp=time.time(), detections=detections, question=clip.question
    )

    with handle.trace() as trace:
        gen_messages = list(llm_stage.process(iter([detection_msg])))
        accumulated = gen_messages[-1].text if gen_messages else ""  # type: ignore[union-attr]

        yn: str | None = None
        if is_yes_no:
            decisions = [
                m
                for m in decision_stage.process(iter(gen_messages))
                if isinstance(m, DecisionMessage)
            ]
            if decisions:
                yn = decisions[0].decision

    return {
        "model": model_name,
        "application": clip.application,
        "clip_id": clip.id,
        "video_file": clip.file,
        "label": clip.label,
        "start_s": clip.start_s,
        "end_s": clip.end_s,
        "expected": clip.expected,
        "run_idx": cycle,
        "n_frames": n_frames,
        "n_detections_total": len(detections),
        "n_unique_labels": len({d.label for d in detections}),
        **_score_clip_response(accumulated, clip, trace, is_yes_no=is_yes_no, yn=yn),
    }


def _detect_and_aggregate(
    handle: PipelineHandle, frames: list[np.ndarray], question: str
) -> list[Detection]:
    """Run this pipeline's detection stage on each frame, then aggregate.

    Args:
        handle: Loaded pipeline whose first stage is an ``ImageDetectionStage``.
        frames: BGR uint8 frames to run detection on.
        question: Question to stamp onto every frame (carried through to the
            aggregated ``DetectionMessage``).

    Returns:
        Aggregated, representative ``Detection`` list for the whole clip.
    """
    detection_stage = handle.stages[0]
    per_frame_msgs: list[DetectionMessage] = []
    for frame in frames:
        msg = RawFrameMessage(frame=frame, timestamp=time.time(), question=question)
        (out,) = list(detection_stage.process(iter([msg])))
        assert isinstance(out, DetectionMessage)  # noqa: S101
        per_frame_msgs.append(out)

    aggregation_stage = DetectionAggregationStage(window=len(frames))
    (aggregated,) = list(aggregation_stage.process(iter(per_frame_msgs)))
    assert isinstance(aggregated, DetectionMessage)  # noqa: S101
    return aggregated.detections


# ---------------------------------------------------------------------------
# Benchmark loops
# ---------------------------------------------------------------------------


def _run_vlm_benchmark(
    handle: PipelineHandle,
    model_name: str,
    clips: list[Clip],
    n_cycles: int,
    data_dir: Path,
    progress: Progress,
    clip_task_id: TaskID,
) -> list[dict]:
    """Run all VLM clips x n_cycles and return result rows.

    Args:
        handle: Loaded pipeline: ``[VLMDescriptionStage, DecisionStage]``.
        model_name: Human-readable name for output rows.
        clips: Clips to evaluate.
        n_cycles: Number of repetitions per clip.
        data_dir: Root data directory; videos are at ``data_dir/videos/<clip.file>``.
        progress: Rich Progress instance.
        clip_task_id: Task ID of the clip sub-progress bar.

    Returns:
        List of result dicts, one per (clip, cycle).
    """
    progress.reset(clip_task_id, total=len(clips) * n_cycles)
    rows: list[dict] = []

    for cycle in range(1, n_cycles + 1):
        for clip in clips:
            progress.update(
                clip_task_id,
                description=f"  {clip.id} [{cycle}/{n_cycles}]",
            )
            try:
                video_path = data_dir / "videos" / clip.file
                raw_frames = _extract_frames_1fps(video_path, clip.start_s, clip.end_s)
                if not raw_frames:
                    console.print(f"  [yellow]{clip.id}: no frames extracted, skipping.[/yellow]")
                    progress.advance(clip_task_id)
                    continue
                resized = [_resize_480p(f) for f in raw_frames]
                row = _run_vlm_clip(handle, model_name, clip, cycle, resized)
            except Exception as exc:  # noqa: BLE001
                console.print(
                    f"  [yellow]{model_name} clip={clip.id} cycle={cycle}: {exc}[/yellow]"
                )
                progress.advance(clip_task_id)
                continue
            rows.append(row)
            progress.advance(clip_task_id)

    return rows


def _run_llm_benchmark(
    handle: PipelineHandle,
    model_name: str,
    clips: list[Clip],
    n_cycles: int,
    data_dir: Path,
    progress: Progress,
    clip_task_id: TaskID,
) -> list[dict]:
    """Run all LLM clips x n_cycles and return result rows.

    For each clip, frames are extracted once and detection is run once; the
    resulting aggregated detections are reused across all cycles.

    Args:
        handle: Loaded pipeline: ``[ImageDetectionStage, LLMStage, DecisionStage]``.
        model_name: Human-readable name for output rows.
        clips: Clips to evaluate.
        n_cycles: Number of repetitions per clip.
        data_dir: Root data directory; videos are at ``data_dir/videos/<clip.file>``.
        progress: Rich Progress instance.
        clip_task_id: Task ID of the clip sub-progress bar.

    Returns:
        List of result dicts, one per (clip, cycle).
    """
    progress.reset(clip_task_id, total=len(clips) * n_cycles)
    rows: list[dict] = []

    for clip in clips:
        try:
            video_path = data_dir / "videos" / clip.file
            raw_frames = _extract_frames_1fps(video_path, clip.start_s, clip.end_s)
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [yellow]{clip.id}: frame extraction failed — {exc}[/yellow]")
            progress.advance(clip_task_id, advance=n_cycles)
            continue

        if not raw_frames:
            console.print(f"  [yellow]{clip.id}: no frames extracted, skipping.[/yellow]")
            progress.advance(clip_task_id, advance=n_cycles)
            continue

        try:
            with handle.trace():
                aggregated = _detect_and_aggregate(handle, raw_frames, clip.question)
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [yellow]{clip.id}: detection failed — {exc}[/yellow]")
            progress.advance(clip_task_id, advance=n_cycles)
            continue

        for cycle in range(1, n_cycles + 1):
            progress.update(
                clip_task_id,
                description=f"  {clip.id} [{cycle}/{n_cycles}]",
            )
            try:
                row = _run_llm_clip(handle, model_name, clip, cycle, aggregated, len(raw_frames))
            except Exception as exc:  # noqa: BLE001
                console.print(
                    f"  [yellow]{model_name} clip={clip.id} cycle={cycle}: {exc}[/yellow]"
                )
                progress.advance(clip_task_id)
                continue
            rows.append(row)
            progress.advance(clip_task_id)

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

    table = Table(title="Real-Data Benchmark Summary", show_lines=True)
    table.add_column("Model", style="bold cyan")
    table.add_column("Kind")
    table.add_column("Load (ms)", justify="right")
    table.add_column("Unload (ms)", justify="right")
    table.add_column("Infer (ms)", justify="right")
    table.add_column("Recall", justify="right", style="bold green")
    table.add_column("YN Acc", justify="right", style="bold yellow")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Mean ITL (ms)", justify="right")

    for model_name, rows in sorted(groups.items()):
        kind = rows[0].get("kind", "?")
        avg_recall = float(np.mean([r["recall"] for r in rows]))
        yn_rows = [r for r in rows if r.get("yn_correct") is not None]
        yn_acc = float(np.mean([r["yn_correct"] for r in yn_rows])) if yn_rows else float("nan")
        ttft_vals = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
        itl_vals = [r["mean_itl_ms"] for r in rows if r.get("mean_itl_ms") is not None]
        load_vals = [r["load_ms"] for r in rows if r.get("load_ms") is not None]
        unload_vals = [r["unload_ms"] for r in rows if r.get("unload_ms") is not None]
        infer_vals = [r["infer_ms"] for r in rows if r.get("infer_ms") is not None]
        table.add_row(
            model_name,
            kind,
            f"{float(np.mean(load_vals)):.0f}" if load_vals else "n/a",
            f"{float(np.mean(unload_vals)):.0f}" if unload_vals else "n/a",
            f"{float(np.mean(infer_vals)):.0f}" if infer_vals else "n/a",
            f"{avg_recall:.3f}",
            f"{yn_acc:.3f}" if yn_rows else "n/a",
            f"{float(np.mean(ttft_vals)):.1f}" if ttft_vals else "n/a",
            f"{float(np.mean(itl_vals)):.1f}" if itl_vals else "n/a",
        )

    console.print(table)


def _write_gzip_results(
    model_entries: list[dict], output_path: Path, *, merge: bool = False
) -> None:
    """Write benchmark results to a gzip-compressed JSON file.

    Entries are matched by ``(model, detector)`` pair (LLM entries) or by
    ``model`` alone (VLM entries, ``detector`` defaults to ``""``).

    Args:
        model_entries: Per-model result dicts with ``load_ms``/``unload_ms`` and ``runs``.
        output_path: Destination ``.json.gz`` path.
        merge: If ``True``, merge into any existing file rather than overwriting.
    """
    existing_models: list[dict] = []
    if merge and output_path.exists():
        try:
            with gzip.open(output_path, "rt", encoding="utf-8") as f:
                existing = json.load(f)
            existing_models = existing.get("models", [])
        except (json.JSONDecodeError, OSError, gzip.BadGzipFile):
            existing_models = []

    def _entry_key(e: dict) -> tuple[str, str]:
        return (e["model"], e.get("detector", ""))

    new_keys = {_entry_key(e) for e in model_entries}
    merged = [e for e in existing_models if _entry_key(e) not in new_keys] + model_entries

    output = {
        "script": "benchmark_real",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "models": merged,
    }
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    default_output = str(Path("bench/results") / f"benchmark_real_{ts}.json.gz")
    parser = argparse.ArgumentParser(
        description="Benchmark VLMs and LLMs against real annotated video clips."
    )
    parser.add_argument(
        "--data-dir",
        default="bench/data",
        help="Directory containing annotations.json and videos/ (default: bench/data).",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=_N_CYCLES,
        help=f"Inference cycles per clip (default: {_N_CYCLES}).",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"Output gzip-compressed JSON path (default: {default_output}).",
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
        "E.g. --models smolvlm2_256m,qwen3_0_6b",
    )
    parser.add_argument(
        "--model-group",
        choices=["vlm", "llm"],
        default=None,
        help="Run only VLMs or only LLMs (overrides --models).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_MAX_TOKENS,
        help=f"Maximum tokens per model response (default: {_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference on CPU instead of GPU.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge results into an existing output file rather than overwriting it.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Entry point for the real-data benchmark script."""
    args = _parse_args()
    n_cycles: int = args.n_cycles
    max_tokens: int = args.max_tokens

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compute_unit = ComputeUnit.CPU if args.cpu else ComputeUnit.GPU

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
    app = Moment2Action(config, qairt=True)

    try:
        clips = _load_clips(data_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    if not clips:
        console.print("[yellow]No clips found in annotations.json. Add clips and retry.[/yellow]")
        sys.exit(0)

    model_filter = set(args.models.split(",")) if args.models else None

    if args.model_group == "vlm":
        vlm_configs = _VLM_CONFIGS[:]
        llm_configs = []
    elif args.model_group == "llm":
        vlm_configs = []
        llm_configs = _LLM_CONFIGS[:]
    else:
        vlm_configs = [c for c in _VLM_CONFIGS if model_filter is None or c[1] in model_filter]
        llm_configs = [c for c in _LLM_CONFIGS if model_filter is None or c[1] in model_filter]

    if not vlm_configs and not llm_configs:
        console.print("[red]No models selected. Exiting.[/red]")
        sys.exit(1)

    apps = sorted({c.application for c in clips})
    console.rule("[bold]M2A Real-Data Benchmark[/bold]")
    console.print(f"  data       : {data_dir}")
    console.print(f"  clips      : {len(clips)}")
    console.print(f"  apps       : {', '.join(apps)}")
    console.print(f"  cycles     : {n_cycles}")
    console.print(f"  tokens     : {max_tokens}")
    console.print(
        f"  detectors  : {', '.join(d for _, d, *_ in _LLM_DETECTORS)} (for LLM pipeline)"
    )
    console.print(f"  output     : {output_path}")
    console.print(f"  vlm models : {', '.join(c[1] for c in vlm_configs)}")
    console.print(f"  llm models : {', '.join(c[1] for c in llm_configs)}")
    console.print(f"  server     : {server_path}:{port}")
    console.print(f"  device     : {'CPU' if args.cpu else 'GPU'}")
    console.print()

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
        total_models = len(vlm_configs) + len(llm_configs) * len(_LLM_DETECTORS)
        model_task = progress.add_task("models", total=total_models)
        clip_task = progress.add_task("  (waiting)", total=None)

        # --- VLM models ---
        for model_id, model_name in vlm_configs:
            if model_id not in MODEL_REGISTRY:
                console.print(f"  [yellow]{model_name} not in registry, skipping.[/yellow]")
                progress.advance(model_task)
                continue

            progress.update(model_task, description=f"{model_name} (vlm)")
            pipeline_name = f"vlm-{model_name}"

            handle: PipelineHandle | None = None
            try:
                handle = (
                    app.new_pipeline(pipeline_name)
                    .add_stage(
                        VLMDescriptionStage,
                        model_id=model_id,
                        model_kwargs={"system_prompt": BENCHMARK_SYSTEM, "max_tokens": max_tokens},
                        grammar=YES_NO_GRAMMAR,
                        compute_unit=compute_unit,
                    )
                    .add_stage(DecisionStage)
                    .build()
                )
                app.load_pipeline(handle)
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [red]{model_name}: failed to load — {exc}[/red]")
                if handle is not None:
                    app.remove_pipeline(handle)
                progress.advance(model_task)
                continue

            rows = _run_vlm_benchmark(
                handle, model_name, clips, n_cycles, data_dir, progress, clip_task
            )

            app.unload_pipeline(handle)
            report = app.metrics_report(handle)
            load_ms, unload_ms = extract_load_unload_ms(report.traces)
            app.remove_pipeline(handle)

            for row in rows:
                row["kind"] = "vlm"
                row["load_ms"] = round(load_ms, 3)
                row["unload_ms"] = round(unload_ms, 3)
            model_entries.append(
                {
                    "model": model_name,
                    "kind": "vlm",
                    "load_ms": round(load_ms, 3),
                    "unload_ms": round(unload_ms, 3),
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

        # --- LLM models (require detector) ---
        if llm_configs:
            for (
                detector_model_id,
                detector_display,
                detector_unit,
                detector_variant,
            ) in _LLM_DETECTORS:
                console.rule(f"[dim]LLM x {detector_display}[/dim]")

                for model_id, model_name, model_template in llm_configs:
                    if model_id not in MODEL_REGISTRY:
                        console.print(f"  [yellow]{model_name} not in registry, skipping.[/yellow]")
                        progress.advance(model_task)
                        continue

                    progress.update(
                        model_task, description=f"{model_name} (llm/{detector_display})"
                    )
                    pipeline_name = f"llm-{model_name}-{detector_display}"

                    handle = None
                    try:
                        handle = (
                            app.new_pipeline(pipeline_name)
                            .add_stage(
                                ImageDetectionStage,
                                model_id=detector_model_id,
                                variant=detector_variant,
                                compute_unit=detector_unit,
                            )
                            .add_stage(
                                LLMStage,
                                model_id=model_id,
                                model_kwargs={
                                    "system_prompt": BENCHMARK_SYSTEM,
                                    "max_tokens": max_tokens,
                                    "template": model_template,
                                },
                                grammar=YES_NO_GRAMMAR,
                                compute_unit=compute_unit,
                            )
                            .add_stage(DecisionStage)
                            .build()
                        )
                        app.load_pipeline(handle)
                    except Exception as exc:  # noqa: BLE001
                        console.print(f"  [red]{model_name}: failed to load — {exc}[/red]")
                        if handle is not None:
                            app.remove_pipeline(handle)
                        progress.advance(model_task)
                        continue

                    rows = _run_llm_benchmark(
                        handle, model_name, clips, n_cycles, data_dir, progress, clip_task
                    )

                    app.unload_pipeline(handle)
                    report = app.metrics_report(handle)
                    llm_class_name = MODEL_REGISTRY[model_id].model_class.__name__
                    load_ms, unload_ms = extract_load_unload_ms(
                        report.traces, name_contains=llm_class_name
                    )
                    app.remove_pipeline(handle)

                    for row in rows:
                        row["kind"] = "llm"
                        row["detector"] = detector_display
                        row["load_ms"] = round(load_ms, 3)
                        row["unload_ms"] = round(unload_ms, 3)
                    model_entries.append(
                        {
                            "model": model_name,
                            "kind": "llm",
                            "detector": detector_display,
                            "load_ms": round(load_ms, 3),
                            "unload_ms": round(unload_ms, 3),
                            "runs": rows,
                        }
                    )

                    if rows:
                        avg_recall = np.mean([r["recall"] for r in rows])
                        console.print(
                            f"  [dim]{model_name}: {len(rows)} results "
                            f"— recall {avg_recall:.3f}[/dim]"
                        )

                    all_rows.extend(rows)
                    progress.advance(model_task)

    if all_rows:
        _write_gzip_results(model_entries, output_path, merge=args.merge)
        console.print(f"\n[green]Results written to {output_path}[/green]")
    else:
        console.print("[yellow]No results produced.[/yellow]")

    console.print()
    _print_summary(all_rows)


if __name__ == "__main__":
    main()
