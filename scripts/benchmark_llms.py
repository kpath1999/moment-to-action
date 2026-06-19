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
"""Benchmark LLM models on application-specific classification prompts.

Each scene maps to one of the five target applications (violence detection,
fall detection, animal threat, eating detection, PPE compliance).  Every
scene poses the binary or multi-label question the deployed system would ask.

Inputs are restricted to what real models actually produce:
  - Detections from YOLO: label, confidence, bounding box (pixel coordinates).
    Spatial context (overlap, orientation, foreground/background) is derived
    from the bboxes rather than assumed from free-form natural language.
  - Audio transcript from an audio model, where the application uses audio.

Two scenes per application: one positive case, one negative case.

Accuracy is keyword recall: words a model produces only by answering the
classification question correctly, not by echoing the input labels.

Usage:
    uv run python scripts/benchmark_llms.py [--n-cycles 3] [--output llm_benchmark_results.csv]

Requires llama_server_path to be set in the M2A config (or pass --server-path).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from moment_to_action.config import AppConfig, load_config
from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models import MODEL_REGISTRY, ModelID, ModelManager
from moment_to_action.models.image.detection._types import BoundingBox, Detection
from moment_to_action.paths import PathManager

console = Console()

# (ModelID, display name) — one backend per model (llama-server handles GPU internally)
_MODEL_CONFIGS: list[tuple[ModelID, str]] = [
    (ModelID.QWEN2_1_5B_INSTRUCT, "qwen2_1_5b"),
    (ModelID.QWEN2_7B_INSTRUCT, "qwen2_7b"),
    (ModelID.QWEN3_4B, "qwen3_4b"),
    (ModelID.PHI35_MINI_INSTRUCT, "phi35_mini"),
]

_N_CYCLES = 3
_MAX_TOKENS = 128

_BENCHMARK_SYSTEM = (
    "You are a scene analysis AI. Answer the user's question directly and concisely. "
    "Lead with your direct answer, then give one sentence of reasoning."
)

# Required PPE items — used to infer what is absent in PPE scenes.
_REQUIRED_PPE: frozenset[str] = frozenset({"hard hat", "safety vest", "glove", "boot"})

# Standard frame dimensions assumed for bbox context derivation.
_FRAME_W = 640
_FRAME_H = 480

# Thresholds for spatial context derivation.
_DEPTH_FG_THRESH = 0.25  # bbox area fraction → foreground
_DEPTH_MG_THRESH = 0.08  # bbox area fraction → midground (else background)
_OVERLAP_THRESH = 0.05  # IoU above this → "overlapping"
_MIN_PAIR = 2  # minimum persons to compute pairwise IoU


# ---------------------------------------------------------------------------
# Spatial helpers — derive context from raw YOLO bbox output
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
# Scene definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scene:
    """One benchmark scene backed by YOLO-realistic inputs.

    Attributes:
        name: Short identifier used in CSV output.
        app: Target application name.
        task: The binary question the system asks (used as prompt suffix).
        detections: YOLO detections (label + confidence + bbox).  Spatial
            context is derived from bboxes by ``_build_prompt``.
        audio_transcript: Transcript from an audio model.  ``None`` for apps
            that do not use audio.
        expected_label: Correct answer token (e.g. "YES", "NO", "COMPLIANT").
        recall_keywords: Words expected from a correct answer.  Labels that
            appear verbatim in ``detections`` are excluded so that input-echoing
            does not inflate recall.
    """

    name: str
    app: str
    task: str
    detections: list[Detection]
    audio_transcript: str | None
    expected_label: str
    recall_keywords: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt builder — derives spatial context from raw bboxes
# ---------------------------------------------------------------------------


def _build_prompt(scene: Scene) -> str:
    """Build a model prompt from YOLO detections and optional audio.

    Spatial features (overlap, orientation, foreground/background) are derived
    from bounding box coordinates rather than assumed from free text.  No
    language appears in the prompt that could not be computed from real YOLO
    output.

    Args:
        scene: Scene definition.

    Returns:
        Formatted prompt string ending with the binary question.
    """
    lines: list[str] = [f"Task: {scene.task}", ""]

    # --- per-detection lines ---
    det_lines: list[str] = []
    for d in scene.detections:
        zone = _frame_zone(d.bbox)
        depth = _depth(d.bbox)
        parts = [f"{d.label} (conf {d.confidence:.2f}, {zone}, {depth}"]
        if d.label == "person" and _is_horizontal(d.bbox):
            parts.append(", horizontal orientation")
        parts.append(")")
        det_lines.append("".join(parts))
    lines.append("Detections:\n" + "\n".join(f"  - {dl}" for dl in det_lines))

    # --- derived pairwise context ---
    persons = [d for d in scene.detections if d.label == "person"]
    animals = [d for d in scene.detections if d.label in ("dog", "cat", "bear", "wolf")]

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

    # --- audio ---
    if scene.audio_transcript is not None:
        lines.append(f"Audio: {scene.audio_transcript}")

    lines.append("")
    lines.append(scene.task)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scenes — 2 per application (positive then negative)
# ---------------------------------------------------------------------------


def _bb(x1: int, y1: int, x2: int, y2: int) -> BoundingBox:
    """Shorthand BoundingBox constructor.

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
    """Shorthand Detection constructor.

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


_SCENES: list[Scene] = [
    # --- Violence Detection -------------------------------------------------
    # Positive: two persons with heavily overlapping bboxes, audio confirms altercation
    Scene(
        name="violence_fight",
        app="violence_detection",
        task="Is a violent incident occurring? Answer YES or NO, then one sentence of reasoning.",
        detections=[
            _det("person", 0.95, 80, 40, 360, 480),
            _det("person", 0.92, 200, 30, 500, 480),  # large overlap with first person
        ],
        audio_transcript="shouting, impact sounds, glass breaking",
        expected_label="YES",
        recall_keywords=["yes", "fight", "violen", "aggress", "altercation", "physical"],
    ),
    # Negative: two persons at opposite sides of frame, no overlap, calm audio
    Scene(
        name="violence_calm",
        app="violence_detection",
        task="Is a violent incident occurring? Answer YES or NO, then one sentence of reasoning.",
        detections=[
            _det("person", 0.93, 10, 50, 200, 480),  # left side
            _det("person", 0.90, 440, 50, 630, 480),  # right side, no overlap
        ],
        audio_transcript="ambient music, quiet conversation, laughter",
        expected_label="NO",
        recall_keywords=["no", "calm", "peaceful", "safe", "non-violent", "normal"],
    ),
    # --- Fall Detection -----------------------------------------------------
    # Positive: person bbox is horizontal (width >> height), located at bottom of frame
    Scene(
        name="fall_detected",
        app="fall_detection",
        task="Has a person fallen? Answer YES or NO, then one sentence of reasoning.",
        detections=[
            _det("person", 0.91, 50, 390, 520, 470),  # horizontal (w=470 > h=80), bottom frame
            _det("chair", 0.74, 300, 200, 500, 400),
        ],
        audio_transcript=None,
        expected_label="YES",
        recall_keywords=["yes", "fall", "fallen", "ground", "floor", "horizontal", "lying"],
    ),
    # Negative: person bbox is vertical (height >> width), centered in frame
    Scene(
        name="fall_standing",
        app="fall_detection",
        task="Has a person fallen? Answer YES or NO, then one sentence of reasoning.",
        detections=[
            _det("person", 0.95, 220, 40, 400, 480),  # vertical (w=180 < h=440), mid-center
            _det("desk", 0.81, 400, 200, 640, 480),
            _det("monitor", 0.78, 460, 60, 620, 260),
        ],
        audio_transcript=None,
        expected_label="NO",
        recall_keywords=["no", "standing", "upright", "vertical", "normal", "not fallen"],
    ),
    # --- Animal Threat / Attack Detection ----------------------------------
    # Positive: dog bbox overlaps heavily with person bbox, audio confirms aggression
    Scene(
        name="animal_threat",
        app="animal_threat_detection",
        task=(
            "Is an animal posing an immediate threat to a person? "
            "Answer YES or NO, then one sentence of reasoning."
        ),
        detections=[
            _det("person", 0.93, 150, 80, 430, 480),
            _det("dog", 0.88, 350, 180, 620, 480),  # overlaps with person bbox
        ],
        audio_transcript="aggressive barking, growling",
        expected_label="YES",
        recall_keywords=["yes", "threat", "danger", "aggress", "attack", "immediate"],
    ),
    # Negative: dog bbox small and far from person (no overlap), calm audio
    Scene(
        name="animal_safe",
        app="animal_threat_detection",
        task=(
            "Is an animal posing an immediate threat to a person? "
            "Answer YES or NO, then one sentence of reasoning."
        ),
        detections=[
            _det("person", 0.94, 80, 50, 380, 480),  # foreground, left
            _det("dog", 0.76, 530, 320, 610, 400),  # small (background), right, no overlap
        ],
        audio_transcript="ambient park sounds, distant barking",
        expected_label="NO",
        recall_keywords=["no", "safe", "distant", "no threat", "away", "not immediate"],
    ),
    # --- Eating Detection (egocentric wearable) ----------------------------
    # Positive: food items dominate foreground (large bbox area = close to camera)
    Scene(
        name="eating_yes",
        app="eating_detection",
        task=(
            "Egocentric view from wearable camera. "
            "Is the wearer currently eating or drinking? "
            "Answer YES or NO, then one sentence of reasoning."
        ),
        detections=[
            _det("fork", 0.89, 240, 300, 400, 440),  # foreground
            _det("sandwich", 0.84, 140, 270, 450, 460),  # foreground
            _det("plate", 0.91, 70, 260, 580, 470),  # large, foreground
            _det("dining table", 0.72, 0, 410, 640, 480),  # background strip
        ],
        audio_transcript=None,
        expected_label="YES",
        recall_keywords=["yes", "eating", "meal", "consuming", "food", "fork"],
    ),
    # Negative: computer peripherals dominate foreground, food present but background
    Scene(
        name="eating_no",
        app="eating_detection",
        task=(
            "Egocentric view from wearable camera. "
            "Is the wearer currently eating or drinking? "
            "Answer YES or NO, then one sentence of reasoning."
        ),
        detections=[
            _det("keyboard", 0.93, 90, 360, 550, 470),  # foreground
            _det("laptop", 0.88, 140, 200, 500, 400),  # midground
            _det("monitor", 0.85, 40, 40, 600, 300),  # large background
            _det("cup", 0.65, 575, 360, 635, 440),  # small, right corner
        ],
        audio_transcript=None,
        expected_label="NO",
        recall_keywords=["no", "working", "typing", "not eating", "computer", "keyboard"],
    ),
    # --- Workplace Safety / PPE Compliance ---------------------------------
    # Positive: all required PPE items detected on or near the person
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
            _det("hard hat", 0.91, 230, 40, 420, 140),  # top of frame, on head
            _det("safety vest", 0.88, 140, 150, 500, 340),
            _det("glove", 0.79, 120, 310, 230, 420),
            _det("glove", 0.77, 410, 310, 520, 420),
            _det("boot", 0.83, 160, 410, 290, 480),
            _det("boot", 0.80, 350, 410, 480, 480),
        ],
        audio_transcript=None,
        expected_label="COMPLIANT",
        recall_keywords=["compliant", "hat", "vest", "glove", "boot", "all", "present"],
    ),
    # Negative: hard hat and gloves absent from detections
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
            # hard hat and gloves absent
        ],
        audio_transcript=None,
        expected_label="NON-COMPLIANT",
        recall_keywords=["non-compliant", "missing", "hat", "glove", "absent", "violation"],
    ),
]


# ---------------------------------------------------------------------------
# Recall metric
# ---------------------------------------------------------------------------


def _recall(response: str, keywords: list[str]) -> float:
    """Compute keyword recall: fraction of expected classification keywords found.

    Args:
        response: Text generated by the model.
        keywords: Words expected from a correct answer (not from the input labels).

    Returns:
        Fraction in [0, 1] of keywords found (case-insensitive).
    """
    if not keywords:
        return 1.0
    resp_lower = response.lower()
    found = sum(1 for kw in keywords if kw.lower() in resp_lower)
    return found / len(keywords)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _run_benchmark(
    model: object,
    model_name: str,
    metrics: MetricsCollector,
    n_cycles: int,
    progress: Progress,
    scene_task_id: object,
) -> list[dict]:
    """Run all scenes x n_cycles through a loaded model, return result rows.

    Calls the model's prepare/run/post_proc interface directly rather than
    going through LlamaServerStage so that the benchmark controls the prompt.

    Args:
        model: Loaded LlamaGGUFModel instance.
        model_name: Human-readable name for output rows.
        metrics: MetricsCollector for span timing.
        n_cycles: Number of repetitions per scene.
        progress: Rich Progress instance for updating the scene sub-bar.
        scene_task_id: Task ID of the scene sub-progress bar.

    Returns:
        List of result dicts, one per (scene, cycle).
    """
    total_steps = len(_SCENES) * n_cycles
    progress.reset(scene_task_id, total=total_steps)  # type: ignore[arg-type]
    rows: list[dict] = []
    for cycle in range(1, n_cycles + 1):
        for scene_idx, scene in enumerate(_SCENES):
            progress.update(
                scene_task_id,  # type: ignore[arg-type]
                description=f"  {scene.name} [{cycle}/{n_cycles}]",
            )
            prompt = _build_prompt(scene)
            t_start = time.perf_counter_ns()
            try:
                with metrics.start_trace():
                    prepared = model.prepare(prompt)  # type: ignore[attr-defined]
                    raw = model.run(prepared)  # type: ignore[attr-defined]
                    response = model.post_proc(raw)[0]  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                console.print(
                    f"  [yellow]{model_name} scene={scene.name} cycle={cycle}: {exc}[/yellow]"
                )
                progress.advance(scene_task_id)  # type: ignore[arg-type]
                continue
            infer_ms = (time.perf_counter_ns() - t_start) / 1e6
            recall = _recall(response, scene.recall_keywords)

            rows.append(
                {
                    "model": model_name,
                    "app": scene.app,
                    "scene": scene.name,
                    "scene_idx": scene_idx,
                    "expected": scene.expected_label,
                    "run": cycle,
                    "infer_ms": round(infer_ms, 3),
                    "response_chars": len(response),
                    "recall": round(recall, 4),
                }
            )
            progress.advance(scene_task_id)  # type: ignore[arg-type]
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_summary(all_rows: list[dict]) -> None:
    """Print a rich summary table with per-model averages.

    Args:
        all_rows: All result rows from the benchmark.
    """
    import numpy as np  # noqa: PLC0415

    groups: dict[str, list[dict]] = {}
    for row in all_rows:
        groups.setdefault(row["model"], []).append(row)

    table = Table(title="LLM Benchmark Summary", show_lines=True)
    table.add_column("Model", style="bold cyan")
    table.add_column("Load (ms)", justify="right")
    table.add_column("Unload (ms)", justify="right")
    table.add_column("Infer (ms)", justify="right")
    table.add_column("Response (chars)", justify="right")
    table.add_column("Recall", justify="right", style="bold green")

    for model_name, rows in sorted(groups.items()):
        load_ms = rows[0].get("load_ms", 0.0)
        unload_ms = rows[0].get("unload_ms", 0.0)
        avg_infer = float(np.mean([r["infer_ms"] for r in rows]))
        avg_chars = float(np.mean([r["response_chars"] for r in rows]))
        avg_recall = float(np.mean([r["recall"] for r in rows]))
        table.add_row(
            model_name,
            f"{load_ms:.0f}",
            f"{unload_ms:.0f}",
            f"{avg_infer:.1f}",
            f"{avg_chars:.0f}",
            f"{avg_recall:.3f}",
        )

    console.print(table)


_CSV_FLOAT_FIELDS = ("infer_ms", "recall", "load_ms", "unload_ms")
_CSV_INT_FIELDS = ("scene_idx", "run", "response_chars")


def _read_existing_csv(path: Path) -> list[dict]:
    """Read an existing results CSV, coercing numeric columns back to numbers.

    Args:
        path: Path to a CSV previously written by this script.

    Returns:
        List of row dicts with numeric fields as ``float``/``int``.
    """
    rows: list[dict] = []
    with path.open(newline="") as f:
        for raw in csv.DictReader(f):
            row = dict(raw)
            for k in _CSV_FLOAT_FIELDS:
                if k in row:
                    row[k] = float(row[k])
            for k in _CSV_INT_FIELDS:
                if k in row:
                    row[k] = int(row[k])
            rows.append(row)
    return rows


def _merge_rows(
    existing_path: Path,
    new_rows: list[dict],
    rerun_models: set[str],
) -> list[dict]:
    """Merge ``new_rows`` into an existing CSV, replacing re-run models.

    Args:
        existing_path: Path to the existing results CSV (may not exist).
        new_rows: Freshly produced rows from this run.
        rerun_models: Model display names that were re-run this session.

    Returns:
        Merged row list.
    """
    if not existing_path.exists():
        return new_rows
    existing = _read_existing_csv(existing_path)
    kept = [r for r in existing if r["model"] not in rerun_models]
    return kept + new_rows


def _write_csv(rows: list[dict], output_path: Path) -> None:
    """Write benchmark result rows to a CSV file.

    Args:
        rows: List of result dicts.
        output_path: Destination CSV path.
    """
    fieldnames = [
        "model",
        "app",
        "scene",
        "scene_idx",
        "expected",
        "run",
        "load_ms",
        "infer_ms",
        "unload_ms",
        "response_chars",
        "recall",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark LLM models on application-specific classification prompts."
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=_N_CYCLES,
        help=f"Inference cycles per scene (default: {_N_CYCLES}).",
    )
    parser.add_argument(
        "--output",
        default="llm_benchmark_results.csv",
        help="Output CSV path (default: llm_benchmark_results.csv).",
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
        "E.g. --models qwen2_1_5b,phi35_mini",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=_MAX_TOKENS,
        help=f"Maximum tokens per model response (default: {_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge results into the existing --output CSV; re-run models replace existing rows.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: C901, PLR0915
    """Entry point for the LLM benchmark script."""
    import numpy as np  # noqa: PLC0415

    global _N_CYCLES, _MAX_TOKENS  # noqa: PLW0603

    args = _parse_args()
    _N_CYCLES = args.n_cycles
    _MAX_TOKENS = args.max_tokens
    output_path = Path(args.output)

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
    console.rule("[bold]M2A LLM Benchmark[/bold]")
    console.print(f"  apps   : {', '.join(apps)}")
    console.print(f"  scenes : {len(_SCENES)}")
    console.print(f"  cycles : {_N_CYCLES}")
    console.print(f"  tokens : {_MAX_TOKENS}")
    console.print(f"  output : {output_path}")
    console.print(f"  models : {', '.join(c[1] for c in configs)}")
    console.print(f"  server : {server_path}:{port}")
    if args.merge:
        console.print("  merge  : on")
    console.print()

    manager = ModelManager(path_manager)
    all_rows: list[dict] = []
    rerun_models: set[str] = set()

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

            rerun_models.add(model_name)
            progress.update(model_task, description=model_name)

            t_load = time.perf_counter_ns()
            try:
                model = manager.get_model(
                    model_id,
                    server_path=config.llama_server_path,
                    port=config.llama_server_port,
                    system_prompt=_BENCHMARK_SYSTEM,
                    max_tokens=_MAX_TOKENS,
                )
                model.load(ComputeBackend(preferred_unit=ComputeUnit.GPU))  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [red]{model_name}: failed to start — {exc}[/red]")
                progress.advance(model_task)
                continue
            load_ms = (time.perf_counter_ns() - t_load) / 1e6
            console.print(f"  [dim]{model_name}: server started in {load_ms:.0f} ms[/dim]")

            metrics = MetricsCollector()
            rows = _run_benchmark(model, model_name, metrics, _N_CYCLES, progress, scene_task)

            t_unload = time.perf_counter_ns()
            try:
                model.unload()  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [yellow]{model_name}: unload error — {exc}[/yellow]")
            unload_ms = (time.perf_counter_ns() - t_unload) / 1e6

            for row in rows:
                row["load_ms"] = round(load_ms, 3)
                row["unload_ms"] = round(unload_ms, 3)

            if rows:
                avg_infer = np.mean([r["infer_ms"] for r in rows])
                avg_recall = np.mean([r["recall"] for r in rows])
                console.print(
                    f"  [dim]{model_name}: {len(rows)} results — "
                    f"avg infer {avg_infer:.0f} ms, recall {avg_recall:.3f}[/dim]"
                )

            all_rows.extend(rows)
            progress.advance(model_task)

    if args.merge:
        all_rows = _merge_rows(output_path, all_rows, rerun_models)

    if all_rows:
        _write_csv(all_rows, output_path)
        console.print(f"\n[green]Results written to {output_path}[/green]")
    else:
        console.print("[yellow]No results produced.[/yellow]")

    console.print()
    _print_summary(all_rows)


if __name__ == "__main__":
    main()
