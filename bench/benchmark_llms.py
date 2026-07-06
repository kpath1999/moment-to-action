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

Each scene (``bench/_scenes.py``) maps to one of the five target applications
(violence detection, fall detection, animal threat, eating detection, PPE
compliance). Every scene poses the binary or multi-label question the deployed
system would ask.

Inputs are restricted to what real models actually produce:
  - Detections from YOLO: label, confidence, bounding box (pixel coordinates).
    Spatial context (overlap, orientation, foreground/background) is derived
    from the bboxes rather than assumed from free-form natural language.
  - Audio transcript from an audio model, where the application uses audio.

Each (model, scene, cycle) is driven through a real
``Pipeline([LLMStage(model, grammar=YES_NO_GRAMMAR), DecisionStage])`` over a
``DetectionMessage`` built from the scene — the same composition an on-device
app would use. Yes/no scenes get the grammar (so ``DecisionStage`` can extract
a verdict); PPE compliance scenes answer COMPLIANT/NON-COMPLIANT and are run
without it, scored on keyword recall only.

Accuracy metrics:
  - ``yn_correct``: bool — whether the extracted decision matched the expected label
    (``None`` for PPE scenes, which have no yes/no verdict).
  - ``recall``: float in [0, 1] — keyword recall for classification keywords.

Timing metrics (streamed from ``MetricsCollector.timed_stream`` via ``LLMStage``):
  - ``ttft_ms``: time from stream start to first token.
  - ``ttfyd_ms``: time from stream start to first yes/no decision.
  - ``mean_itl_ms``, ``std_itl_ms``: inter-token latency statistics.
  - ``inference_metrics``: llama.cpp-native timing fields from the stop chunk.

Usage:
    uv run python bench/benchmark_llms.py [--n-cycles 3] [--output results.json]

Requires llama_server_path to be set in the M2A config (or pass --server-path).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from _common import build_context, console, write_results
from _scenes import SCENES, Scene
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from moment_to_action.benchmarking import extract_load_unload_ms, recall
from moment_to_action.config import AppConfig
from moment_to_action.hardware import ComputeUnit
from moment_to_action.messages import DecisionMessage, DetectionMessage, GenerationMessage
from moment_to_action.metrics import SpanType
from moment_to_action.models import MODEL_REGISTRY, ModelID
from moment_to_action.prompting import BENCHMARK_SYSTEM, CHATML, PHI3, YES_NO_GRAMMAR
from moment_to_action.stages.llm import DecisionStage, LLMStage

if TYPE_CHECKING:
    from moment_to_action.metrics import MetricsCollector, Span
    from moment_to_action.models.llm._base import LlamaGGUFModel

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

_YES_NO_LABELS = frozenset({"yes", "no"})


# ---------------------------------------------------------------------------
# Streaming benchmark
# ---------------------------------------------------------------------------


def _run_scene(
    model: LlamaGGUFModel, model_name: str, scene: Scene, cycle: int, metrics: MetricsCollector
) -> dict:
    """Drive one scene through ``Pipeline([LLMStage, DecisionStage])`` and score it.

    Yes/no scenes get ``grammar=YES_NO_GRAMMAR`` so the decision is forced to
    the first token and ``DecisionStage`` can extract it; PPE compliance
    scenes (expecting COMPLIANT/NON-COMPLIANT) run without the grammar and are
    scored on keyword recall only. ``metrics.timed_stream`` (inside
    ``LLMStage``) records ``ttft_ms``/``ttfyd_ms``/``mean_itl_ms``/``std_itl_ms``
    onto the model's ``MODEL_INFERENCE`` span as tokens arrive, independent of
    whether the caller drains the full response — this benchmark drains fully
    so ``recall`` can be scored against complete text, while still capturing
    an accurate ``ttfyd_ms`` (the time an early-abort consumer would have saved).

    Note: ``scene.audio_transcript`` is not yet wired in — ``LLMStage`` builds its
    prompt from ``DetectionMessage.detections`` only, with no extra-context hook.

    Args:
        model: Loaded LlamaGGUFModel instance.
        model_name: Display name for the result row.
        scene: Scene to evaluate.
        cycle: Cycle index (1-based).
        metrics: The same MetricsCollector *model* was constructed with.

    Returns:
        Result dict for this (scene, cycle).
    """
    is_yes_no = scene.expected_label.lower() in _YES_NO_LABELS
    grammar = YES_NO_GRAMMAR if is_yes_no else None
    detection_msg = DetectionMessage(
        timestamp=time.time(), detections=scene.detections, question=scene.task
    )

    llm_stage = LLMStage(model, grammar=grammar, metrics=metrics)
    gen_messages = list(llm_stage.process(iter([detection_msg])))
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


def _run_benchmark(
    model: LlamaGGUFModel, model_name: str, n_cycles: int, metrics: MetricsCollector
) -> list[dict]:
    """Run all scenes x n_cycles through a loaded model, return result rows.

    Args:
        model: Loaded LlamaGGUFModel instance.
        model_name: Human-readable name for output rows.
        n_cycles: Number of repetitions per scene.
        metrics: The same MetricsCollector *model* was constructed with.

    Returns:
        List of result dicts, one per (scene, cycle).
    """
    rows: list[dict] = []
    for cycle in range(1, n_cycles + 1):
        for scene in SCENES:
            try:
                rows.append(_run_scene(model, model_name, scene, cycle, metrics))
            except Exception as exc:  # noqa: BLE001, PERF203
                console.print(
                    f"  [yellow]{model_name} scene={scene.name} cycle={cycle}: {exc}[/yellow]"
                )
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
    table.add_column("Scenes", justify="right")
    table.add_column("Recall", justify="right", style="bold green")
    table.add_column("YN Acc", justify="right")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("TTFYD (ms)", justify="right")

    for model_name, rows in sorted(groups.items()):
        recalls = [r["recall"] for r in rows]
        yn_rows = [r for r in rows if r["yn_correct"] is not None]
        yn_acc = np.mean([r["yn_correct"] for r in yn_rows]) if yn_rows else float("nan")
        ttft_vals = [r["ttft_ms"] for r in rows if r["ttft_ms"] is not None]
        ttfyd_vals = [r["ttfyd_ms"] for r in rows if r["ttfyd_ms"] is not None]
        table.add_row(
            model_name,
            str(len(rows)),
            f"{np.mean(recalls):.3f}",
            f"{yn_acc:.3f}" if yn_rows else "n/a",
            f"{np.mean(ttft_vals):.1f}" if ttft_vals else "n/a",
            f"{np.mean(ttfyd_vals):.1f}" if ttfyd_vals else "n/a",
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
    args = _parse_args()
    n_cycles: int = args.n_cycles
    max_tokens: int = args.max_tokens
    output_path = Path(args.output)
    compute_unit = ComputeUnit.CPU if args.cpu else ComputeUnit.GPU

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
    console.rule("[bold]M2A LLM Benchmark[/bold]")
    console.print(f"  apps   : {', '.join(apps)}")
    console.print(f"  scenes : {len(SCENES)}")
    console.print(f"  cycles : {n_cycles}")
    console.print(f"  tokens : {max_tokens}")
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

        for model_id, model_name, model_template in configs:
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
                    template=model_template,
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

                rows = _run_benchmark(model, model_name, n_cycles, ctx.metrics)  # type: ignore[arg-type]

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
            script="benchmark_llms",
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
