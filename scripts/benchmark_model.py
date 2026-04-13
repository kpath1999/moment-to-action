"""Profile a single model across CPU, GPU, and NPU compute units.

For each requested compute unit, loads the model, runs warmup iterations,
then collects latency percentiles, peak memory, power draw, and model size.
Results are printed as a Rich table and optionally saved to JSON.

Usage::

    uv run python scripts/benchmark_model.py --model yolo
    uv run python scripts/benchmark_model.py --model mobileclip --units cpu npu
    uv run python scripts/benchmark_model.py --model smolvlm2 --units cpu --n-runs 10
    uv run python scripts/benchmark_model.py --model qwen3 --units cpu --output qwen3_profile.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import rich
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from moment_to_action.benchmark import (
    BenchmarkConfig,
    MobileCLIPBenchmark,
    ModelBenchmark,
    Qwen3Benchmark,
    SmolVLM2Benchmark,
    VariantID,
    VariantProfile,
    VariantRegistry,
    YOLOBenchmark,
)
from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.models import ModelID, ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=Console(stderr=True))],
)
logger = logging.getLogger(__name__)

# ── CLI ────────────────────────────────────────────────────────────────────────

_MODEL_CHOICES: dict[str, tuple[ModelID, ModelBenchmark]] = {
    "yolo": (ModelID.YOLO_V8, YOLOBenchmark()),
    "mobileclip": (ModelID.MOBILECLIP_S2, MobileCLIPBenchmark()),
    "smolvlm2": (ModelID.SMOLVLM2_2_2B, SmolVLM2Benchmark()),
    "qwen3": (ModelID.QWEN3_4B, Qwen3Benchmark()),
}

_UNIT_MAP: dict[str, ComputeUnit] = {
    "cpu": ComputeUnit.CPU,
    "gpu": ComputeUnit.GPU,
    "npu": ComputeUnit.NPU,
}

parser = argparse.ArgumentParser(
    description="Profile a model across compute units and report latency, memory, and cost metrics."
)
parser.add_argument(
    "--model",
    required=True,
    choices=list(_MODEL_CHOICES),
    help="Which model to benchmark.",
)
parser.add_argument(
    "--units",
    nargs="+",
    choices=list(_UNIT_MAP),
    default=["cpu", "gpu", "npu"],
    metavar="UNIT",
    help="Compute units to profile (default: cpu gpu npu).",
)
parser.add_argument(
    "--n-warmup",
    type=int,
    default=5,
    metavar="N",
    help="Number of warmup inference passes (default: 5).",
)
parser.add_argument(
    "--n-runs",
    type=int,
    default=20,
    metavar="N",
    help="Number of timed inference passes per unit (default: 20).",
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    metavar="FILE",
    help="Optional path to write JSON results.",
)
parser.add_argument(
    "--save-registry",
    action="store_true",
    help="Persist results into the default VariantRegistry cache.",
)
args = parser.parse_args()

# ── Setup ──────────────────────────────────────────────────────────────────────

model_id, benchmark = _MODEL_CHOICES[args.model]
manager = ModelManager()
config = BenchmarkConfig(n_warmup=args.n_warmup, n_runs=args.n_runs, batch_sizes=[1])
registry = VariantRegistry()
profiles: list[VariantProfile] = []

# ── Profile each unit ─────────────────────────────────────────────────────────

for unit_name in args.units:
    unit = _UNIT_MAP[unit_name]
    logger.info("─" * 60)
    logger.info("Profiling %s on %s …", args.model.upper(), unit_name.upper())

    backend = ComputeBackend(preferred_unit=unit)
    actual_unit = backend.active_unit

    if actual_unit != unit:
        logger.warning(
            "  %s not available — fell back to %s, skipping.",
            unit_name.upper(),
            actual_unit.name,
        )
        continue

    try:
        profile = benchmark.profile(backend=backend, manager=manager, config=config)
    except Exception:
        logger.exception("  Failed to profile on %s — skipping.", unit_name.upper())
        continue

    registry.register(profile)
    profiles.append(profile)
    logger.info(
        "  Done: mean=%.1fms  p95=%.1fms  mem=%.0fMB",
        profile.inference_mean_ms,
        profile.inference_p95_ms,
        profile.peak_memory_mb,
    )

logger.info("─" * 60)

if not profiles:
    logger.error("No profiles collected — check that the model is downloaded and units are valid.")
    sys.exit(1)

# ── Results table ─────────────────────────────────────────────────────────────

table = Table(title=f"Benchmark: {args.model.upper()} ({model_id.value})", show_lines=True)
table.add_column("Unit", style="bold cyan", justify="center")
table.add_column("Load (ms)", justify="right")
table.add_column("Mean (ms)", justify="right")
table.add_column("p50 (ms)", justify="right")
table.add_column("p95 (ms)", justify="right")
table.add_column("p99 (ms)", justify="right")
table.add_column("Peak Mem (MB)", justify="right")
table.add_column("Max Batch", justify="right")
table.add_column("Power (mW)", justify="right")
table.add_column("Energy/inf (mJ)", justify="right")
table.add_column("Model Size (MB)", justify="right")

for p in profiles:
    unit_str = p.variant_id.compute_unit.value
    power = f"{p.cost.power_mw:.0f}" if p.cost.power_mw is not None else "—"
    energy = f"{p.cost.energy_per_inference_mj:.3f}" if p.cost.energy_per_inference_mj is not None else "—"
    table.add_row(
        unit_str,
        f"{p.load_latency_ms:.1f}",
        f"{p.inference_mean_ms:.2f}",
        f"{p.inference_p50_ms:.2f}",
        f"{p.inference_p95_ms:.2f}",
        f"{p.inference_p99_ms:.2f}",
        f"{p.peak_memory_mb:.0f}",
        str(p.max_batch_size),
        power,
        energy,
        f"{p.model_size_bytes / (1024 ** 2):.1f}",
    )

rich.print(table)

# Best-variant summary
best_latency = registry.best_variant(model_id, "latency")
if best_latency:
    rich.print(
        f"\n[bold green]Fastest unit:[/] {best_latency.variant_id.compute_unit.value} "
        f"@ {best_latency.inference_mean_ms:.2f}ms mean"
    )

best_efficiency = registry.best_variant(model_id, "efficiency")
if best_efficiency and best_efficiency.cost.energy_per_inference_mj is not None:
    rich.print(
        f"[bold green]Most efficient unit:[/] {best_efficiency.variant_id.compute_unit.value} "
        f"@ {best_efficiency.cost.energy_per_inference_mj:.3f}mJ/inference"
    )

# ── Persist ───────────────────────────────────────────────────────────────────

if args.save_registry:
    registry.save()
    logger.info("Registry saved to %s", registry.path)

if args.output is not None:
    payload = {
        "model": args.model,
        "model_id": model_id.value,
        "config": {
            "n_warmup": config.n_warmup,
            "n_runs": config.n_runs,
        },
        "profiles": [p.json() for p in profiles],
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Results written to %s", args.output)
