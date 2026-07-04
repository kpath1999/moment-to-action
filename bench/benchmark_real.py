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

VLM pipeline:
  1. Extract frames at 1 FPS from the annotated ROI window.
  2. Resize each frame to at most 480 px tall (CPU image tower constraint).
  3. Encode as base64 JPEG and pass all frames to the VLM.
  4. Stream the response and collect full timing + accuracy metrics.

LLM pipeline:
  1. Extract frames at 1 FPS from the annotated ROI window.
  2. Run a detection model (YOLO V8 or Detectron2) on each frame via
     ``prepare`` / ``run`` / ``post_proc``.  Detection spans are sub-spans
     within the LLM model's MetricsCollector.
  3. Aggregate detections across all frames (keep highest-confidence instance
     per label; record frame count per label).
  4. Build a structured text prompt from the aggregated detections (spatial
     context derived from bboxes, same helpers as benchmark_llms.py).
  5. Stream the LLM response and collect full timing + accuracy metrics.

Model lists at the top of this file — comment out any entry to skip it.

Usage:
    uv run python bench/benchmark_real.py [options]

Requires ``llama_server_path`` to be set in the M2A config (or pass ``--server-path``).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import gzip
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
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

from moment_to_action.config import AppConfig, load_config
from moment_to_action.hardware import ComputeUnit, Platform
from moment_to_action.hardware._loaded_models._llama import LlamaModel
from moment_to_action.metrics import MetricsCollector, SpanType
from moment_to_action.models import MODEL_REGISTRY, ModelID, ModelManager
from moment_to_action.paths import PathManager

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moment_to_action.hardware._metrics import LlamaCppInferenceMetrics
    from moment_to_action.metrics._types import MetricsReport
    from moment_to_action.models.image.detection._types import BoundingBox, Detection

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)


@contextlib.contextmanager
def _silence_native_output() -> Iterator[None]:
    """Redirect OS-level stdout+stderr to /dev/null for the duration of the block.

    The QAIRT runtime emits C++ chatter (e.g. "Profile Logger with name = defaultKey
    doesn't exist!") straight to file descriptors 1/2, bypassing Python's logging and
    corrupting the rich progress bar.

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
# Model lists — comment out any entry to skip it
# ---------------------------------------------------------------------------

# Prompt templates for LLM chat formats.
_CHATML = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_PHI3 = "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"

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
    (ModelID.QWEN3_0_6B, "qwen3_0_6b", _CHATML),
    (ModelID.QWEN3_1_7B, "qwen3_1_7b", _CHATML),
    (ModelID.GEMMA3_270M_IT, "gemma3_270m", _CHATML),
    (ModelID.GEMMA3_1B_IT, "gemma3_1b", _CHATML),
    (ModelID.QWEN2_1_5B_INSTRUCT, "qwen2_1_5b", _CHATML),
    (ModelID.QWEN3_4B, "qwen3_4b", _CHATML),
    (ModelID.PHI35_MINI_INSTRUCT, "phi35_mini", _PHI3),
]

# Detectors used for the LLM pipeline — comment out any to skip.
# Tuple: (ModelID, display_name, ComputeUnit, variant_key)
_LLM_DETECTORS: list[tuple[ModelID, str, ComputeUnit, str]] = [
    # (ModelID.YOLO_V8, "yolo_v8", ComputeUnit.NPU, "qcs6490"),
    (ModelID.DETECTRON2, "detectron2", ComputeUnit.NPU, "qcs6490_w8a16"),
]

_N_CYCLES = 3
_MAX_TOKENS = 128

_BENCHMARK_SYSTEM = (
    "You are a scene analysis AI. Answer the user's question directly and concisely. "
    "Lead with your direct answer, then give one sentence of reasoning."
)

# Standard assumed frame dimensions for spatial context derivation.
_FRAME_W = 640
_FRAME_H = 480

# Thresholds for spatial context derivation (same as benchmark_llms.py).
_DEPTH_FG_THRESH = 0.25
_DEPTH_MG_THRESH = 0.08
_OVERLAP_THRESH = 0.05
_MIN_PAIR = 2
_MAX_FRAME_HEIGHT = 480

# All COCO animal classes (used for person-animal proximity context in prompts).
_COCO_ANIMALS: frozenset[str] = frozenset(
    ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe")
)


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
        ValueError: If the JSON is malformed or clips are missing required fields.
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
    import cv2  # noqa: PLC0415

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
    import cv2  # noqa: PLC0415

    h, w = frame.shape[:2]
    if h <= _MAX_FRAME_HEIGHT:
        return frame
    scale = _MAX_FRAME_HEIGHT / h
    return cv2.resize(frame, (int(w * scale), _MAX_FRAME_HEIGHT), interpolation=cv2.INTER_AREA)


def _bgr_to_b64(frame: np.ndarray) -> str:
    """Convert a BGR uint8 frame to a base64-encoded JPEG string.

    Args:
        frame: BGR uint8 image array.

    Returns:
        Base64-encoded JPEG bytes as a UTF-8 string (no ``data:`` prefix).
    """
    import cv2  # noqa: PLC0415

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Spatial helpers (ported from benchmark_llms.py)
# ---------------------------------------------------------------------------


def _area(b: BoundingBox) -> float:
    """Compute bounding box area in pixels.

    Args:
        b: Bounding box.

    Returns:
        Area in pixels.
    """
    return (b.x2 - b.x1) * (b.y2 - b.y1)


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    """Compute intersection-over-union between two bounding boxes.

    Args:
        a: First bounding box.
        b: Second bounding box.

    Returns:
        IoU in [0, 1].
    """
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _frame_zone(b: BoundingBox) -> str:
    """Return a natural-language frame zone for a bounding box centroid.

    Args:
        b: Bounding box.

    Returns:
        String like "bottom-left", "mid-center", etc.
    """
    cx = (b.x1 + b.x2) / 2
    cy = (b.y1 + b.y2) / 2
    h = "left" if cx < _FRAME_W / 3 else ("right" if cx > 2 * _FRAME_W / 3 else "center")
    v = "top" if cy < _FRAME_H / 3 else ("bottom" if cy > 2 * _FRAME_H / 3 else "mid")
    return f"{v}-{h}"


def _depth(b: BoundingBox) -> str:
    """Return foreground/midground/background based on bbox area fraction.

    Args:
        b: Bounding box.

    Returns:
        "foreground", "midground", or "background".
    """
    frac = _area(b) / (_FRAME_W * _FRAME_H)
    if frac > _DEPTH_FG_THRESH:
        return "foreground"
    if frac > _DEPTH_MG_THRESH:
        return "midground"
    return "background"


def _is_horizontal(b: BoundingBox) -> bool:
    """Return True when the bounding box is wider than it is tall.

    Args:
        b: Bounding box.

    Returns:
        True if width > height.
    """
    return (b.x2 - b.x1) > (b.y2 - b.y1)


# ---------------------------------------------------------------------------
# Detection aggregation and prompt building
# ---------------------------------------------------------------------------


def _aggregate_detections(per_frame: list[list[Detection]]) -> list[Detection]:
    """Aggregate per-frame detections into a single representative set.

    For each unique label, keeps the instance with the highest confidence
    across all frames.

    Args:
        per_frame: List of detection lists, one per extracted frame.

    Returns:
        List of representative Detection objects, one per unique label.
    """
    best: dict[str, Detection] = {}
    for frame_dets in per_frame:
        for det in frame_dets:
            if det.label not in best or det.confidence > best[det.label].confidence:
                best[det.label] = det
    return list(best.values())


def _build_llm_prompt(clip: Clip, detections: list[Detection]) -> str:
    """Build a structured text prompt from aggregated detections.

    Spatial features (overlap, orientation, foreground/background) are derived
    from bounding box coordinates.

    Args:
        clip: Clip being evaluated (provides the task question).
        detections: Aggregated representative detections from the video ROI.

    Returns:
        Formatted prompt string ending with the binary question.
    """
    lines: list[str] = [f"Task: {clip.question}", ""]

    det_lines: list[str] = []
    for d in detections:
        zone = _frame_zone(d.bbox)
        dep = _depth(d.bbox)
        parts = [f"{d.label} (conf {d.confidence:.2f}, {zone}, {dep}"]
        if d.label == "person" and _is_horizontal(d.bbox):
            parts.append(", horizontal orientation")
        parts.append(")")
        det_lines.append("".join(parts))
    lines.append("Detections:\n" + "\n".join(f"  - {dl}" for dl in det_lines))

    persons = [d for d in detections if d.label == "person"]
    animals = [d for d in detections if d.label in _COCO_ANIMALS]

    if len(persons) >= _MIN_PAIR:
        max_person_iou = max(
            _iou(persons[i].bbox, persons[j].bbox)
            for i in range(len(persons))
            for j in range(i + 1, len(persons))
        )
        overlap_desc = "overlapping" if max_person_iou > _OVERLAP_THRESH else "non-overlapping"
        lines.append(f"Person bounding boxes: {overlap_desc} (max IoU={max_person_iou:.2f})")

    if persons and animals:
        max_pa_iou = max(_iou(p.bbox, a.bbox) for p in persons for a in animals)
        pa_desc = "overlapping with person" if max_pa_iou > _OVERLAP_THRESH else "not overlapping"
        lines.append(f"Animal bounding box: {pa_desc} (max IoU with person={max_pa_iou:.2f})")

    lines.append("")
    lines.append(clip.question)
    return "\n".join(lines)


def _build_payload(
    prompt: str,
    max_tokens: int,
    system_prompt: str,
    template: str | None,
) -> dict:
    """Build a llama.cpp /completion request dict for text-only inference.

    Args:
        prompt: User prompt text.
        max_tokens: Maximum tokens to generate.
        system_prompt: System message text.
        template: Optional format string with ``{system}`` / ``{user}`` placeholders.

    Returns:
        ``/completion`` request body dict.
    """
    if template is not None:
        full_prompt = template.format(system=system_prompt, user=prompt)
    else:
        full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt
    return {"prompt": full_prompt, "n_predict": max_tokens}


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _recall(response: str, keywords: list[str]) -> float:
    """Compute keyword recall: fraction of expected keywords found in the response.

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
    """Return "yes" or "no" if the response contains a yes/no decision, else ``None``.

    Args:
        text: Accumulated model response so far.

    Returns:
        "yes", "no", or ``None`` if not yet decidable.
    """
    cleaned = text.strip().lower()
    words = cleaned.split()
    if words and words[0].rstrip(".,!?;:") in {"yes", "no"}:
        return words[0].rstrip(".,!?;:")
    m = re.search(r"\banswer\s*:\s*(yes|no)\b", cleaned)
    if m:
        return m.group(1)
    return None


def _extract_load_unload_ms(report: MetricsReport, name_contains: str = "") -> tuple[float, float]:
    """Extract load and unload latencies from a completed MetricsReport.

    When the trace contains spans from multiple models (e.g. detector + LLM),
    pass *name_contains* to restrict matching to spans whose name includes that
    substring (e.g. ``"LlamaGGUFModel"``).

    Args:
        report: A completed MetricsReport.
        name_contains: Optional substring filter on span name.

    Returns:
        Tuple of (load_ms, unload_ms); 0.0 for any span not found.
    """
    load_ms = 0.0
    unload_ms = 0.0
    for trace in report.traces:
        for span in trace.spans:
            if name_contains and name_contains not in (span.name or ""):
                continue
            if span.type_ == SpanType.MODEL_LOAD:
                load_ms = span.latency_ms
            elif span.type_ == SpanType.MODEL_UNLOAD:
                unload_ms = span.latency_ms
    return load_ms, unload_ms


# ---------------------------------------------------------------------------
# VLM scene runner
# ---------------------------------------------------------------------------


def _run_vlm_clip(
    model: object,
    model_name: str,
    clip: Clip,
    cycle: int,
    b64_frames: list[str],
    frame_h: int,
    frame_w: int,
    metrics: MetricsCollector,
) -> dict:
    """Stream one clip through a VLM and collect all metrics.

    Args:
        model: Loaded LlamaVLModel instance.
        model_name: Display name for the result.
        clip: Clip being evaluated.
        cycle: Cycle index (1-based).
        b64_frames: Base64-encoded JPEG frames for this clip's ROI.
        frame_h: Height of each frame after 480p resize.
        frame_w: Width of each frame after 480p resize.
        metrics: Active MetricsCollector (a trace must be open).

    Returns:
        Result dict for this (clip, cycle).

    Raises:
        TypeError: If ``model._loaded_model`` is not a LlamaModel.
    """
    loaded_model = getattr(model, "_loaded_model", None)
    if not isinstance(loaded_model, LlamaModel):
        msg = f"{model_name}: _loaded_model is not a LlamaModel"
        raise TypeError(msg)

    prepared = model.prepare((clip.question, b64_frames))  # type: ignore[attr-defined]

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
        yn_correct = yn == clip.expected.lower()

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
        "n_frames": len(b64_frames),
        "frame_w": frame_w,
        "frame_h": frame_h,
        "response": accumulated,
        "response_chars": len(accumulated),
        "yn_correct": yn_correct,
        "recall": round(_recall(accumulated, clip.recall_keywords), 4),
        "infer_ms": round(infer_ms, 3),
        "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
        "ttfyd_ms": round(ttfyd_ms, 3) if ttfyd_ms is not None else None,
        "mean_itl_ms": round(mean_itl, 3),
        "std_itl_ms": round(std_itl, 3),
        "inference_metrics": inf_m.model_dump() if inf_m is not None else None,
    }


# ---------------------------------------------------------------------------
# LLM scene runner
# ---------------------------------------------------------------------------


def _run_llm_clip(  # noqa: PLR0913
    model: object,
    model_name: str,
    clip: Clip,
    cycle: int,
    detections: list[Detection],
    n_frames: int,
    max_tokens: int,
    system_prompt: str,
    template: str | None,
    metrics: MetricsCollector,
) -> dict:
    """Stream one clip through an LLM and collect all metrics.

    The prompt is built from aggregated detections; no images are passed.

    Args:
        model: Loaded LlamaGGUFModel instance.
        model_name: Display name for the result.
        clip: Clip being evaluated.
        cycle: Cycle index (1-based).
        detections: Aggregated representative detections from the clip's ROI.
        n_frames: Number of frames that were extracted (for metadata).
        max_tokens: Maximum tokens to generate.
        system_prompt: System message for the prompt.
        template: Optional chat template with ``{system}``/``{user}`` placeholders.
        metrics: Active MetricsCollector (a trace must be open).

    Returns:
        Result dict for this (clip, cycle).

    Raises:
        TypeError: If ``model._loaded_model`` is not a LlamaModel.
    """
    loaded_model = getattr(model, "_loaded_model", None)
    if not isinstance(loaded_model, LlamaModel):
        msg = f"{model_name}: _loaded_model is not a LlamaModel"
        raise TypeError(msg)

    prompt = _build_llm_prompt(clip, detections)
    payload = _build_payload(prompt, max_tokens, system_prompt, template)

    t0 = time.perf_counter_ns()
    ttft_ms: float | None = None
    ttfyd_ms: float | None = None
    token_times: list[int] = []
    accumulated = ""
    inf_m: LlamaCppInferenceMetrics | None = None

    with metrics.start_span(SpanType.MODEL_INFERENCE, f"{model_name}.stream") as span:
        for token in loaded_model.stream(payload):
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
        yn_correct = yn == clip.expected.lower()

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
        "n_detections_total": sum(len(f) for f in [detections]),
        "n_unique_labels": len({d.label for d in detections}),
        "prompt": prompt,
        "response": accumulated,
        "response_chars": len(accumulated),
        "yn_correct": yn_correct,
        "recall": round(_recall(accumulated, clip.recall_keywords), 4),
        "infer_ms": round(infer_ms, 3),
        "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
        "ttfyd_ms": round(ttfyd_ms, 3) if ttfyd_ms is not None else None,
        "mean_itl_ms": round(mean_itl, 3),
        "std_itl_ms": round(std_itl, 3),
        "inference_metrics": inf_m.model_dump() if inf_m is not None else None,
    }


# ---------------------------------------------------------------------------
# Detection helper (used in LLM pipeline)
# ---------------------------------------------------------------------------


def _detect_frames(
    detector: object,
    frames: list[np.ndarray],
    metrics: MetricsCollector,
    detector_name: str,
) -> list[list[Detection]]:
    """Run the detector on each frame and return per-frame detection lists.

    Each frame's prepare/run/post_proc calls are wrapped in sub-spans of the
    provided MetricsCollector.

    Args:
        detector: Loaded ImageDetectionModel instance.
        frames: BGR uint8 frames to run detection on.
        metrics: Active MetricsCollector; detection spans are sub-spans.
        detector_name: Display name used in span labels.

    Returns:
        List of Detection lists, one per input frame.
    """
    results: list[list[Detection]] = []
    for i, frame in enumerate(frames):
        span_name = f"{detector_name}.detect.frame{i}"
        with metrics.start_span(SpanType.MODEL_INFERENCE, span_name):
            try:
                with _silence_native_output():
                    prepared = detector.prepare(frame)  # type: ignore[attr-defined]
                    raw = detector.run(prepared)  # type: ignore[attr-defined]
                    dets: list[Detection] = detector.post_proc(raw)  # type: ignore[attr-defined]
                results.append(dets)
            except Exception:  # noqa: BLE001
                results.append([])
    return results


# ---------------------------------------------------------------------------
# Benchmark loops
# ---------------------------------------------------------------------------


def _run_vlm_benchmark(
    model: object,
    model_name: str,
    clips: list[Clip],
    n_cycles: int,
    data_dir: Path,
    metrics: MetricsCollector,
    progress: Progress,
    clip_task_id: TaskID,
) -> list[dict]:
    """Run all VLM clips x n_cycles and return result rows.

    Args:
        model: Loaded LlamaVLModel instance.
        model_name: Human-readable name for output rows.
        clips: Clips to evaluate.
        n_cycles: Number of repetitions per clip.
        data_dir: Root data directory; videos are at ``data_dir/videos/<clip.file>``.
        metrics: Active MetricsCollector (a trace must be open).
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
                b64_frames = [_bgr_to_b64(f) for f in resized]
                frame_h, frame_w = resized[0].shape[:2]
                row = _run_vlm_clip(
                    model, model_name, clip, cycle, b64_frames, frame_h, frame_w, metrics
                )
            except Exception as exc:  # noqa: BLE001
                console.print(
                    f"  [yellow]{model_name} clip={clip.id} cycle={cycle}: {exc}[/yellow]"
                )
                progress.advance(clip_task_id)
                continue
            rows.append(row)
            progress.advance(clip_task_id)

    return rows


def _run_llm_benchmark(  # noqa: PLR0913
    model: object,
    model_name: str,
    model_template: str | None,
    clips: list[Clip],
    n_cycles: int,
    data_dir: Path,
    detector: object,
    detector_name: str,
    metrics: MetricsCollector,
    progress: Progress,
    clip_task_id: TaskID,
) -> list[dict]:
    """Run all LLM clips x n_cycles and return result rows.

    For each clip, frames are extracted once and detection is run once; the
    resulting aggregated detections are reused across all cycles.  Detection
    spans are sub-spans within ``metrics``.

    Args:
        model: Loaded LlamaGGUFModel instance.
        model_name: Human-readable name for output rows.
        model_template: Optional chat template string.
        clips: Clips to evaluate.
        n_cycles: Number of repetitions per clip.
        data_dir: Root data directory; videos are at ``data_dir/videos/<clip.file>``.
        detector: Loaded ImageDetectionModel instance for the LLM pipeline.
        detector_name: Display name for the detector (used in span labels).
        metrics: Active MetricsCollector (a trace must be open).
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

        per_frame = _detect_frames(detector, raw_frames, metrics, detector_name)
        aggregated = _aggregate_detections(per_frame)

        for cycle in range(1, n_cycles + 1):
            progress.update(
                clip_task_id,
                description=f"  {clip.id} [{cycle}/{n_cycles}]",
            )
            try:
                row = _run_llm_clip(
                    model,
                    model_name,
                    clip,
                    cycle,
                    aggregated,
                    len(raw_frames),
                    _MAX_TOKENS,
                    _BENCHMARK_SYSTEM,
                    model_template,
                    metrics,
                )
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
        itl_vals = [r["mean_itl_ms"] for r in rows]
        load_vals = [r["load_ms"] for r in rows if r.get("load_ms") is not None]
        unload_vals = [r["unload_ms"] for r in rows if r.get("unload_ms") is not None]
        infer_vals = [r["infer_ms"] for r in rows]
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


def _write_json(
    model_entries: list[dict],
    output_path: Path,
    *,
    merge: bool = False,
) -> None:
    """Write benchmark results to a JSON file.

    Output is written gzip-compressed with no indentation to keep file size down.

    When *merge* is ``True`` and *output_path* already exists, entries for
    models present in *model_entries* (matched by ``(model, detector)`` pair for
    LLM entries and by ``model`` alone for VLM entries) are replaced; others are
    preserved.

    Args:
        model_entries: Per-model result dicts with ``metrics_report`` and ``runs``.
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
    global _N_CYCLES, _MAX_TOKENS  # noqa: PLW0603

    args = _parse_args()
    _N_CYCLES = args.n_cycles
    _MAX_TOKENS = args.max_tokens

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compute_unit = ComputeUnit.CPU if args.cpu else ComputeUnit.GPU

    path_manager = PathManager()
    config = load_config(path_manager.app_config_file)

    from moment_to_action.qairt._manager import QairtSDKManager  # noqa: PLC0415

    QairtSDKManager.from_app_config(config, path_manager).configure_env()

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
    console.print(f"  cycles     : {_N_CYCLES}")
    console.print(f"  tokens     : {_MAX_TOKENS}")
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

            platform = Platform(config)
            metrics = MetricsCollector(platform)
            manager = ModelManager(path_manager, metrics=metrics)
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
                    model.load(platform, compute_unit)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [red]{model_name}: failed to load — {exc}[/red]")
                    progress.advance(model_task)
                    continue

                rows = _run_vlm_benchmark(
                    model, model_name, clips, _N_CYCLES, data_dir, metrics, progress, clip_task
                )

                try:
                    model.unload()
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [yellow]{model_name}: unload error — {exc}[/yellow]")

            report = metrics.report()
            load_ms, unload_ms = _extract_load_unload_ms(report)
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

        # --- LLM models (require detector) ---
        if llm_configs:
            for (
                detector_model_id,
                detector_display,
                detector_unit,
                detector_variant,
            ) in _LLM_DETECTORS:
                console.rule(f"[dim]LLM x {detector_display}[/dim]")

                # Load detector once and reuse across all LLM models.
                detector_manager = ModelManager(path_manager)
                try:
                    detector_obj = detector_manager.get_model(
                        detector_model_id, variant=detector_variant
                    )
                except Exception as exc:  # noqa: BLE001
                    console.print(
                        f"  [red]Detector {detector_display}: failed to get — {exc}[/red]"
                    )
                    for _ in llm_configs:
                        progress.advance(model_task)
                    continue

                detector_platform = Platform(config)
                try:
                    with _silence_native_output():
                        detector_obj.load(detector_platform, detector_unit)
                except Exception as exc:  # noqa: BLE001
                    console.print(
                        f"  [red]Detector {detector_display}: failed to load — {exc}[/red]"
                    )
                    for _ in llm_configs:
                        progress.advance(model_task)
                    continue

                for model_id, model_name, model_template in llm_configs:
                    if model_id not in MODEL_REGISTRY:
                        console.print(f"  [yellow]{model_name} not in registry, skipping.[/yellow]")
                        progress.advance(model_task)
                        continue

                    progress.update(
                        model_task, description=f"{model_name} (llm/{detector_display})"
                    )

                    platform = Platform(config)
                    metrics = MetricsCollector(platform)
                    manager = ModelManager(path_manager, metrics=metrics)
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

                    rows = []
                    with metrics.start_trace():
                        try:
                            model.load(platform, compute_unit)
                        except Exception as exc:  # noqa: BLE001
                            console.print(f"  [red]{model_name}: failed to load — {exc}[/red]")
                            progress.advance(model_task)
                            continue

                        rows = _run_llm_benchmark(
                            model,
                            model_name,
                            model_template,
                            clips,
                            _N_CYCLES,
                            data_dir,
                            detector_obj,
                            detector_display,
                            metrics,
                            progress,
                            clip_task,
                        )

                        try:
                            model.unload()
                        except Exception as exc:  # noqa: BLE001
                            console.print(f"  [yellow]{model_name}: unload error — {exc}[/yellow]")

                    report = metrics.report()
                    load_ms, unload_ms = _extract_load_unload_ms(report)
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
                            "metrics_report": report.json(),
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

                try:
                    with _silence_native_output():
                        detector_obj.unload()
                except Exception as exc:  # noqa: BLE001
                    console.print(
                        f"  [yellow]Detector {detector_display} unload error — {exc}[/yellow]"
                    )

    if all_rows:
        _write_json(model_entries, output_path, merge=args.merge)
        console.print(f"\n[green]Results written to {output_path}[/green]")
    else:
        console.print("[yellow]No results produced.[/yellow]")

    console.print()
    _print_summary(all_rows)


if __name__ == "__main__":
    main()
