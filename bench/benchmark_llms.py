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

Accuracy metrics:
  - ``yn_correct``: bool — whether the model's first word was the correct yes/no.
  - ``recall``: float in [0, 1] — keyword recall for classification keywords.

Timing metrics (streaming-derived):
  - ``ttft_ms``: time from stream start to first token.
  - ``ttfyd_ms``: time from stream start to first yes/no decision.
  - ``mean_itl_ms``, ``std_itl_ms``: inter-token latency statistics.
  - ``inference_metrics``: llama.cpp-native timing fields from the stop chunk.

Usage:
    uv run python scripts/benchmark_llms.py [--n-cycles 3] [--output results.json]

Requires llama_server_path to be set in the M2A config (or pass --server-path).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
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

from moment_to_action.benchmarking import detect_yn, extract_load_unload_ms, recall
from moment_to_action.config import AppConfig, load_config
from moment_to_action.hardware import ComputeUnit, Platform
from moment_to_action.hardware._loaded_models._llama import LlamaModel
from moment_to_action.metrics import MetricsCollector, SpanType
from moment_to_action.models import MODEL_REGISTRY, ModelID, ModelManager
from moment_to_action.models.image.detection._types import BoundingBox, Detection
from moment_to_action.paths import PathManager
from moment_to_action.prompting import BENCHMARK_SYSTEM as _BENCHMARK_SYSTEM
from moment_to_action.prompting import CHATML, PHI3
from moment_to_action.prompting import build_detection_prompt as _build_detection_prompt
from moment_to_action.prompting import build_payload as _build_payload

if TYPE_CHECKING:
    from moment_to_action.hardware._metrics import LlamaCppInferenceMetrics

console = Console()

# (ModelID, display name, prompt template | None)
_MODEL_CONFIGS: list[tuple[ModelID, str, str | None]] = [
    (ModelID.QWEN2_1_5B_INSTRUCT, "qwen2_1_5b", CHATML),
    (ModelID.QWEN2_7B_INSTRUCT, "qwen2_7b", CHATML),
    (ModelID.QWEN3_4B, "qwen3_4b", CHATML),
    (ModelID.PHI35_MINI_INSTRUCT, "phi35_mini", PHI3),
    (ModelID.MOONDREAM2, "moondream2", CHATML),
]

_N_CYCLES = 3
_MAX_TOKENS = 128

# Required PPE items — used to infer what is absent in PPE scenes.
_REQUIRED_PPE: frozenset[str] = frozenset({"hard hat", "safety vest", "glove", "boot"})


# ---------------------------------------------------------------------------
# Scene definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scene:
    """One benchmark scene backed by YOLO-realistic inputs.

    Attributes:
        name: Short identifier used in output.
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

    Thin wrapper over :func:`~moment_to_action.prompting.build_detection_prompt`
    that inserts the scene's audio transcript (if any) as an extra context line.

    Args:
        scene: Scene definition.

    Returns:
        Formatted prompt string ending with the binary question.
    """
    extra_lines = (
        [f"Audio: {scene.audio_transcript}"] if scene.audio_transcript is not None else None
    )
    return _build_detection_prompt(scene.detections, scene.task, extra_lines=extra_lines)


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
# Streaming benchmark
# ---------------------------------------------------------------------------


def _run_scene(
    model: object,
    model_name: str,
    scene: Scene,
    cycle: int,
    max_tokens: int,
    system_prompt: str,
    template: str | None,
    metrics: MetricsCollector,
) -> dict:
    """Stream one scene through the model and collect all metrics.

    Wraps the streaming loop in a MODEL_INFERENCE span.  Within the span,
    tracks TTFT, TTFYD, and inter-token latencies manually via perf_counter,
    then stores them as span metadata via ``metrics.set_meta``.

    Args:
        model: Loaded LlamaGGUFModel or LlamaVLModel instance.
        model_name: Display name for the result.
        scene: Scene to evaluate.
        cycle: Cycle index (1-based).
        max_tokens: Maximum tokens to generate.
        system_prompt: System message for the prompt.
        template: Optional chat template (``{system}``/``{user}`` placeholders).
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

    prompt = _build_prompt(scene)
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
            if ttfyd_ms is None and detect_yn(accumulated) is not None:
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

    yn = detect_yn(accumulated)
    yn_correct: bool | None = None
    if yn is not None:
        yn_correct = yn == scene.expected_label.lower()

    return {
        "model": model_name,
        "app": scene.app,
        "scene": scene.name,
        "expected": scene.expected_label,
        "run_idx": cycle,
        "response": accumulated,
        "response_chars": len(accumulated),
        "yn_correct": yn_correct,
        "recall": round(recall(accumulated, scene.recall_keywords), 4),
        "infer_ms": round(infer_ms, 3),
        "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
        "ttfyd_ms": round(ttfyd_ms, 3) if ttfyd_ms is not None else None,
        "mean_itl_ms": round(mean_itl, 3),
        "std_itl_ms": round(std_itl, 3),
        "inference_metrics": inf_m.model_dump() if inf_m is not None else None,
    }


def _run_benchmark(  # noqa: PLR0913
    model: object,
    model_name: str,
    metrics: MetricsCollector,
    n_cycles: int,
    max_tokens: int,
    system_prompt: str,
    template: str | None,
    progress: Progress,
    scene_task_id: object,
) -> list[dict]:
    """Run all scenes x n_cycles through a loaded model, return result rows.

    Args:
        model: Loaded LlamaGGUFModel instance.
        model_name: Human-readable name for output rows.
        metrics: Active MetricsCollector (a trace must be open).
        n_cycles: Number of repetitions per scene.
        max_tokens: Maximum tokens to generate per scene.
        system_prompt: System message for each prompt.
        template: Optional chat template (``{system}``/``{user}`` placeholders).
        progress: Rich Progress instance for updating the scene sub-bar.
        scene_task_id: Task ID of the scene sub-progress bar.

    Returns:
        List of result dicts, one per (scene, cycle).
    """
    total_steps = len(_SCENES) * n_cycles
    progress.reset(scene_task_id, total=total_steps)  # type: ignore[arg-type]
    rows: list[dict] = []
    for cycle in range(1, n_cycles + 1):
        for scene in _SCENES:
            progress.update(
                scene_task_id,  # type: ignore[arg-type]
                description=f"  {scene.name} [{cycle}/{n_cycles}]",
            )
            try:
                row = _run_scene(
                    model, model_name, scene, cycle, max_tokens, system_prompt, template, metrics
                )
            except Exception as exc:  # noqa: BLE001
                console.print(
                    f"  [yellow]{model_name} scene={scene.name} cycle={cycle}: {exc}[/yellow]"
                )
                progress.advance(scene_task_id)  # type: ignore[arg-type]
                continue
            rows.append(row)
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
    groups: dict[str, list[dict]] = {}
    for row in all_rows:
        groups.setdefault(row["model"], []).append(row)

    table = Table(title="LLM Benchmark Summary", show_lines=True)
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


def _write_json(model_entries: list[dict], output_path: Path, *, merge: bool = False) -> None:
    """Write benchmark results to a JSON file.

    When *merge* is ``True`` and *output_path* already exists, entries for models
    present in *model_entries* are replaced and entries for models not in the current
    run are preserved.  When *merge* is ``False`` (default) the file is overwritten.

    Each entry in ``model_entries`` contains a single model's ``metrics_report``
    (once, not duplicated per row) plus a ``runs`` list of per-scene result dicts.

    Args:
        model_entries: Per-model result dicts, each containing ``metrics_report`` and ``runs``.
        output_path: Destination JSON path.
        merge: If ``True``, merge into any existing file rather than overwriting.
    """
    existing_models: list[dict] = []
    if merge and output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            existing_models = existing.get("models", [])
        except (json.JSONDecodeError, OSError):
            existing_models = []

    new_names = {e["model"] for e in model_entries}
    merged = [e for e in existing_models if e["model"] not in new_names] + model_entries

    output = {
        "script": "benchmark_llms",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "models": merged,
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
        default="llm_benchmark_results.json",
        help="Output JSON path (default: llm_benchmark_results.json).",
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


def main() -> None:  # noqa: C901, PLR0915
    """Entry point for the LLM benchmark script."""
    global _N_CYCLES, _MAX_TOKENS  # noqa: PLW0603

    args = _parse_args()
    _N_CYCLES = args.n_cycles
    _MAX_TOKENS = args.max_tokens
    output_path = Path(args.output)
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
    console.print(f"  device : {'CPU' if args.cpu else 'GPU'}")
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
        model_task = progress.add_task("models", total=len(configs))
        scene_task = progress.add_task("  (waiting)", total=None)

        for model_id, model_name, model_template in configs:
            if model_id not in MODEL_REGISTRY:
                console.print(f"  [yellow]{model_name} not in registry, skipping.[/yellow]")
                progress.advance(model_task)
                continue

            progress.update(model_task, description=model_name)

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
                    console.print(f"  [red]{model_name}: failed to start — {exc}[/red]")
                    progress.advance(model_task)
                    continue

                rows = _run_benchmark(
                    model,
                    model_name,
                    metrics,
                    _N_CYCLES,
                    _MAX_TOKENS,
                    _BENCHMARK_SYSTEM,
                    model_template,
                    progress,
                    scene_task,
                )

                try:
                    model.unload()
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [yellow]{model_name}: unload error — {exc}[/yellow]")

            report = metrics.report()
            load_ms, unload_ms = extract_load_unload_ms(report)
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
        _write_json(model_entries, output_path, merge=args.merge)
        console.print(f"\n[green]Results written to {output_path}[/green]")
    else:
        console.print("[yellow]No results produced.[/yellow]")

    console.print()
    _print_summary(all_rows)


if __name__ == "__main__":
    main()
