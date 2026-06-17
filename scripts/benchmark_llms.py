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
"""Benchmark LLM models on scene-reasoning prompts.

Creates a fixed set of synthetic scenes and asks each model a reasoning question
that cannot be answered by repeating the input (e.g. "Is this safe? What should
the person do?").  Accuracy is measured as keyword recall against action/context
terms that a model can only produce by actually reasoning about the scene — not
by echoing the object list.

Usage:
    uv run python scripts/benchmark_llms.py [--n-cycles 3] [--output llm_benchmark_results.csv]

Requires llama_server_path to be set in the M2A config (or pass --server-path).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
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

# System prompt: steer toward recommendation, not description.
_BENCHMARK_SYSTEM = (
    "You are a scene analyst. Given detected objects, briefly describe what is "
    "most likely happening and recommend one specific action. Respond in 1-2 sentences. "
    "Do not just list the objects."
)


def _bb(x1: int, y1: int, x2: int, y2: int) -> BoundingBox:
    """Shorthand BoundingBox constructor for scene definitions."""
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _det(label: str, conf: float, x1: int, y1: int, x2: int, y2: int) -> Detection:
    """Shorthand Detection constructor for scene definitions."""
    return Detection(label=label, confidence=conf, bbox=_bb(x1, y1, x2, y2))


def _build_benchmark_prompt(detections: list[Detection]) -> str:
    """Build a reasoning prompt from a detection list.

    Presents only object labels (no raw pixel coordinates or confidence scores —
    they add noise that small models cannot reason over) and asks for a situational
    assessment + recommendation.

    Args:
        detections: Detections from a synthetic scene.

    Returns:
        Formatted prompt string.
    """
    top5 = sorted(detections, key=lambda d: d.confidence, reverse=True)[:5]
    labels = ", ".join(d.label for d in top5)
    return (
        f"Objects detected: {labels}.\nWhat is most likely happening and what should the person do?"
    )


# Each scene: (name, detections, action_keywords).
# action_keywords are terms a model should produce when reasoning correctly —
# these cannot be satisfied by simply echoing the detection list.
_SCENES: list[tuple[str, list[Detection], list[str]]] = [
    (
        "kitchen_prep",
        [
            _det("person", 0.95, 50, 10, 200, 480),
            _det("refrigerator", 0.88, 300, 0, 500, 480),
            _det("cup", 0.72, 210, 200, 260, 250),
            _det("bottle", 0.65, 270, 180, 310, 260),
        ],
        # expect reasoning about food/drink preparation
        ["cook", "prepare", "food", "drink", "eat", "kitchen", "meal", "water"],
    ),
    (
        "street_traffic",
        [
            _det("person", 0.93, 10, 50, 120, 480),
            _det("car", 0.91, 200, 200, 600, 420),
            _det("truck", 0.79, 620, 150, 900, 450),
            _det("traffic light", 0.84, 350, 10, 390, 80),
        ],
        # expect traffic-safety reasoning
        ["wait", "cross", "look", "traffic", "caution", "safe", "signal", "road", "danger"],
    ),
    (
        "office_work",
        [
            _det("person", 0.97, 100, 0, 300, 480),
            _det("laptop", 0.90, 310, 200, 540, 380),
            _det("chair", 0.76, 80, 300, 200, 480),
            _det("monitor", 0.82, 550, 100, 800, 350),
        ],
        # expect work/desk reasoning
        ["work", "sit", "type", "computer", "desk", "office", "screen", "task"],
    ),
    (
        "relaxing_tv",
        [
            _det("person", 0.89, 20, 100, 200, 480),
            _det("couch", 0.92, 150, 280, 700, 480),
            _det("tv", 0.85, 250, 50, 600, 270),
            _det("remote", 0.61, 400, 300, 450, 330),
        ],
        # expect leisure/relaxation reasoning
        ["watch", "sit", "relax", "rest", "television", "couch", "channel", "leisure"],
    ),
    (
        "outdoor_activity",
        [
            _det("person", 0.93, 50, 20, 220, 480),
            _det("bicycle", 0.87, 250, 150, 550, 480),
            _det("backpack", 0.73, 10, 50, 120, 200),
            _det("dog", 0.68, 560, 300, 720, 480),
        ],
        # expect cycling/outdoor activity reasoning
        ["ride", "cycle", "exercise", "outdoor", "trail", "park", "sport", "walk"],
    ),
    (
        "fall_emergency",
        [
            _det("person", 0.91, 50, 380, 500, 480),  # person near floor (y1 high = near bottom)
            _det("cell phone", 0.78, 520, 420, 600, 480),
            _det("chair", 0.65, 300, 100, 500, 380),
        ],
        # expect emergency/help reasoning — person near floor suggests a fall
        ["help", "call", "emergency", "fallen", "assist", "911", "medical", "floor", "injured"],
    ),
    (
        "crowded_transit",
        [
            _det("person", 0.96, 10, 0, 180, 480),
            _det("person", 0.94, 200, 20, 380, 480),
            _det("suitcase", 0.88, 390, 200, 600, 480),
            _det("backpack", 0.82, 610, 100, 750, 380),
            _det("person", 0.79, 760, 0, 900, 480),
        ],
        # expect travel/transit reasoning
        ["travel", "airport", "station", "luggage", "crowd", "transit", "commute", "board"],
    ),
]


def _recall(response: str, expected_keywords: list[str]) -> float:
    """Compute keyword recall: fraction of expected action/context keywords in the response.

    Args:
        response: Text generated by the model.
        expected_keywords: Action or context terms expected from correct reasoning.

    Returns:
        Fraction in [0, 1] of expected keywords found (case-insensitive).
    """
    if not expected_keywords:
        return 1.0
    resp_lower = response.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in resp_lower)
    return found / len(expected_keywords)


def _run_benchmark(
    model: object,
    model_name: str,
    metrics: MetricsCollector,
) -> list[dict]:
    """Run all scenes x N_CYCLES through a loaded model, return rows.

    Calls the model's prepare/run/post_proc interface directly rather than going
    through LlamaServerStage so that the benchmark can use its own reasoning
    prompt instead of the stage's description prompt.

    Args:
        model: Loaded LlamaGGUFModel instance.
        model_name: Human-readable name for output rows.
        metrics: MetricsCollector for span timing.

    Returns:
        List of result dicts, one per (scene, cycle).
    """
    rows: list[dict] = []
    for cycle in range(1, _N_CYCLES + 1):
        for scene_idx, (scene_name, detections, expected_kws) in enumerate(_SCENES):
            prompt = _build_benchmark_prompt(detections)
            t_start = time.perf_counter_ns()
            try:
                with metrics.start_trace():
                    prepared = model.prepare(prompt)  # type: ignore[attr-defined]
                    raw = model.run(prepared)  # type: ignore[attr-defined]
                    response = model.post_proc(raw)[0]  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                console.print(
                    f"  [yellow]{model_name} scene={scene_name} cycle={cycle}: {exc}[/yellow]"
                )
                continue
            infer_ms = (time.perf_counter_ns() - t_start) / 1e6
            recall = _recall(response, expected_kws)

            rows.append(
                {
                    "model": model_name,
                    "scene": scene_name,
                    "scene_idx": scene_idx,
                    "run": cycle,
                    "infer_ms": round(infer_ms, 3),
                    "response_chars": len(response),
                    "recall": round(recall, 4),
                }
            )
    return rows


def _print_summary(all_rows: list[dict]) -> None:
    """Print a rich summary table with averages per model.

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
        avg_infer = np.mean([r["infer_ms"] for r in rows])
        avg_chars = np.mean([r["response_chars"] for r in rows])
        avg_recall = np.mean([r["recall"] for r in rows])
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
        "scene",
        "scene_idx",
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


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Benchmark LLM models on scene-reasoning prompts.")
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
        "--merge",
        action="store_true",
        help="Merge results into the existing --output CSV; re-run models replace existing rows.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: C901, PLR0915
    """Entry point for the LLM benchmark script."""
    import numpy as np  # noqa: PLC0415

    global _N_CYCLES  # noqa: PLW0603

    args = _parse_args()
    _N_CYCLES = args.n_cycles
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

    console.rule("[bold]M2A LLM Benchmark[/bold]")
    console.print(f"  scenes : {len(_SCENES)}")
    console.print(f"  cycles : {_N_CYCLES}")
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

        for model_id, model_name in configs:
            if model_id not in MODEL_REGISTRY:
                console.print(f"  [yellow]{model_name} not in registry, skipping.[/yellow]")
                progress.advance(model_task)
                continue

            rerun_models.add(model_name)
            progress.update(model_task, description=model_name)

            # --- load (start llama-server) ---
            t_load = time.perf_counter_ns()
            try:
                model = manager.get_model(
                    model_id,
                    server_path=config.llama_server_path,
                    port=config.llama_server_port,
                    system_prompt=_BENCHMARK_SYSTEM,
                    max_tokens=128,
                )
                model.load(ComputeBackend(preferred_unit=ComputeUnit.GPU))  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [red]{model_name}: failed to start — {exc}[/red]")
                progress.advance(model_task)
                continue
            load_ms = (time.perf_counter_ns() - t_load) / 1e6
            console.print(f"  [dim]{model_name}: server started in {load_ms:.0f} ms[/dim]")

            metrics = MetricsCollector()
            rows = _run_benchmark(model, model_name, metrics)

            # --- unload (stop llama-server) ---
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
