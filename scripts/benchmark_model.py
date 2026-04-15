"""Profile a single model across CPU, GPU, and NPU compute units.

For each requested compute unit, loads the model, runs warmup iterations,
then collects latency percentiles, peak memory, power draw, model size,
and accuracy metrics when the benchmark provides them. Results are printed as
a Rich table, saved under a timestamped results directory, and can also be
written to the variant registry or an explicit JSON file.

Usage::

    uv run python scripts/benchmark_model.py --model yolo
    uv run python scripts/benchmark_model.py --model mobileclip --units cpu npu
    uv run python scripts/benchmark_model.py --model grounding-dino --units cpu
    uv run python scripts/benchmark_model.py --model siglip --oracle-path logs/oracle.json
"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from moment_to_action.benchmark import (
    BenchmarkConfig,
    GroundingDINOBenchmark,
    MobileCLIPBenchmark,
    ModelBenchmark,
    OracleStore,
    Qwen3Benchmark,
    SigLIPBenchmark,
    SmolVLM2Benchmark,
    VariantProfile,
    VariantRegistry,
    YOLOBenchmark,
)
from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.models import ModelID, ModelManager

faulthandler.enable()

stdout_console = Console()
stderr_console = Console(stderr=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=stderr_console)],
)
logger = logging.getLogger(__name__)

_UNIT_MAP: dict[str, ComputeUnit] = {
    "cpu": ComputeUnit.CPU,
    "gpu": ComputeUnit.GPU,
    "npu": ComputeUnit.NPU,
}

_MODEL_NAMES = (
    "yolo",
    "mobileclip",
    "smolvlm2",
    "qwen3",
    "grounding-dino",
    "siglip",
)

parser = argparse.ArgumentParser(
    description=(
        "Profile a model across compute units and report latency, memory, "
        "accuracy, and cost metrics."
    )
)
parser.add_argument(
    "--model",
    required=True,
    choices=list(_MODEL_NAMES),
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
    "--results-dir",
    type=Path,
    default=None,
    metavar="DIR",
    help=(
        "Directory to write JSON, CSV, and plot PNGs. Defaults to scripts/tmp_results_<timestamp>/."
    ),
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    metavar="FILE",
    help="Additional path to write JSON results.",
)
parser.add_argument(
    "--save-registry",
    action="store_true",
    help="Persist results into the default VariantRegistry cache.",
)
parser.add_argument(
    "--oracle-path",
    type=Path,
    default=None,
    metavar="FILE",
    help=(
        "Path to the oracle ground truth JSON file. Defaults to the platform cache directory. "
        "Oracle models (grounding-dino, siglip) write here."
    ),
)
parser.add_argument(
    "--no-plot",
    action="store_true",
    help="Skip generating bar-chart PNG plots.",
)
parser.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    metavar="LEVEL",
    help="Logging verbosity (default: INFO). Use DEBUG to diagnose delegate crashes.",
)
args = parser.parse_args()

logging.getLogger().setLevel(args.log_level)

scripts_dir = Path(__file__).parent
if args.results_dir is not None:
    results_dir = args.results_dir
else:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    results_dir = scripts_dir / f"tmp_results__{args.model}_{timestamp}"
results_dir.mkdir(parents=True, exist_ok=True)
logger.info("Results will be saved to %s", results_dir)

model_choices: dict[str, tuple[ModelID, ModelBenchmark]] = {
    "yolo": (ModelID.YOLO_V8, YOLOBenchmark()),
    "mobileclip": (ModelID.MOBILECLIP_S2, MobileCLIPBenchmark()),
    "smolvlm2": (ModelID.SMOLVLM2_2_2B, SmolVLM2Benchmark()),
    "qwen3": (ModelID.QWEN2_5_4B, Qwen3Benchmark()),
    "grounding-dino": (ModelID.GROUNDING_DINO_BASE, GroundingDINOBenchmark()),
    "siglip": (ModelID.SIGLIP_SO400M, SigLIPBenchmark()),
}

model_id, benchmark = model_choices[args.model]
if args.oracle_path is not None:
    oracle_store = OracleStore(path=args.oracle_path)
    if args.model == "grounding-dino":
        benchmark = GroundingDINOBenchmark(oracle_store=oracle_store)
    elif args.model == "siglip":
        benchmark = SigLIPBenchmark(oracle_store=oracle_store)

manager = ModelManager()
config = BenchmarkConfig(n_warmup=args.n_warmup, n_runs=args.n_runs, batch_sizes=[1])
registry = VariantRegistry()
profiles: list[VariantProfile] = []

for unit_name in args.units:
    unit = _UNIT_MAP[unit_name]
    logger.info("─" * 60)
    logger.info("Profiling %s on %s …", args.model.upper(), unit_name.upper())

    backend = ComputeBackend(preferred_unit=unit)
    actual_unit = backend.active_unit

    if actual_unit != unit:
        logger.warning(
            "  %s not available -- fell back to %s, skipping.",
            unit_name.upper(),
            actual_unit.name,
        )
        continue

    try:
        profile = benchmark.profile(backend=backend, manager=manager, config=config)
    except Exception:
        logger.exception("  Failed to profile on %s -- skipping.", unit_name.upper())
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
    logger.error("No profiles collected -- check that the model is downloaded and units are valid.")
    sys.exit(1)

table = Table(title=f"Benchmark: {args.model.upper()} ({model_id.value})", show_lines=True)
table.add_column("Unit", style="bold cyan", justify="center")
table.add_column("Load (ms)", justify="right")
table.add_column("Mean (ms)", justify="right")
table.add_column("p50 (ms)", justify="right")
table.add_column("p95 (ms)", justify="right")
table.add_column("p99 (ms)", justify="right")
table.add_column("Accuracy", justify="right")
table.add_column("Peak Mem (MB)", justify="right")
table.add_column("Max Batch", justify="right")
table.add_column("Power (mW)", justify="right")
table.add_column("Energy/inf (mJ)", justify="right")
table.add_column("Model Size (MB)", justify="right")

for profile in profiles:
    unit_str = profile.variant_id.compute_unit.value
    power = f"{profile.cost.power_mw:.0f}" if profile.cost.power_mw is not None else "—"
    energy = (
        f"{profile.cost.energy_per_inference_mj:.3f}"
        if profile.cost.energy_per_inference_mj is not None
        else "—"
    )
    accuracy = f"{profile.accuracy:.3f}" if profile.accuracy is not None else "—"
    table.add_row(
        unit_str,
        f"{profile.load_latency_ms:.1f}",
        f"{profile.inference_mean_ms:.2f}",
        f"{profile.inference_p50_ms:.2f}",
        f"{profile.inference_p95_ms:.2f}",
        f"{profile.inference_p99_ms:.2f}",
        accuracy,
        f"{profile.peak_memory_mb:.0f}",
        str(profile.max_batch_size),
        power,
        energy,
        f"{profile.model_size_bytes / (1024**2):.1f}",
    )

stdout_console.print(table)

best_latency = registry.best_variant(model_id, "latency")
if best_latency:
    stdout_console.print(
        f"\n[bold green]Fastest unit:[/] {best_latency.variant_id.compute_unit.value} "
        f"@ {best_latency.inference_mean_ms:.2f}ms mean"
    )

best_efficiency = registry.best_variant(model_id, "efficiency")
if best_efficiency and best_efficiency.cost.energy_per_inference_mj is not None:
    stdout_console.print(
        f"[bold green]Most efficient unit:[/] {best_efficiency.variant_id.compute_unit.value} "
        f"@ {best_efficiency.cost.energy_per_inference_mj:.3f}mJ/inference"
    )

if args.save_registry:
    registry.save()
    logger.info("Registry saved to %s", registry.path)

payload = {
    "model": args.model,
    "model_id": model_id.value,
    "config": {
        "n_warmup": config.n_warmup,
        "n_runs": config.n_runs,
    },
    "profiles": [profile.json() for profile in profiles],
}

json_path = results_dir / f"{args.model}_profiles.json"
json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
logger.info("JSON results written to %s", json_path)

if args.output is not None:
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("JSON also written to %s", args.output)

csv_path = results_dir / f"{args.model}_profiles.csv"
csv_fields = [
    "model",
    "model_id",
    "compute_unit",
    "hardware_target",
    "load_latency_ms",
    "inference_mean_ms",
    "inference_p50_ms",
    "inference_p95_ms",
    "inference_p99_ms",
    "accuracy",
    "peak_memory_mb",
    "max_batch_size",
    "power_mw",
    "energy_per_inference_mj",
    "model_size_bytes",
    "n_runs",
    "profiled_at",
]
with csv_path.open("w", newline="", encoding="utf-8") as file_handle:
    writer = csv.DictWriter(file_handle, fieldnames=csv_fields)
    writer.writeheader()
    for profile in profiles:
        writer.writerow(
            {
                "model": args.model,
                "model_id": profile.variant_id.model_id.value,
                "compute_unit": profile.variant_id.compute_unit.value,
                "hardware_target": profile.hardware_target,
                "load_latency_ms": profile.load_latency_ms,
                "inference_mean_ms": profile.inference_mean_ms,
                "inference_p50_ms": profile.inference_p50_ms,
                "inference_p95_ms": profile.inference_p95_ms,
                "inference_p99_ms": profile.inference_p99_ms,
                "accuracy": profile.accuracy if profile.accuracy is not None else "",
                "peak_memory_mb": profile.peak_memory_mb,
                "max_batch_size": profile.max_batch_size,
                "power_mw": profile.cost.power_mw if profile.cost.power_mw is not None else "",
                "energy_per_inference_mj": (
                    profile.cost.energy_per_inference_mj
                    if profile.cost.energy_per_inference_mj is not None
                    else ""
                ),
                "model_size_bytes": profile.model_size_bytes,
                "n_runs": profile.n_runs,
                "profiled_at": profile.profiled_at.isoformat(),
            }
        )
logger.info("CSV table written to %s", csv_path)

if not args.no_plot:
    try:
        from plot_benchmark import plot_profiles

        plot_profiles(profiles, model_name=args.model, output_dir=results_dir)
        logger.info("Plots written to %s", results_dir)
    except ImportError:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "plot_benchmark",
            scripts_dir / "plot_benchmark.py",
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            module.plot_profiles(  # type: ignore[attr-defined]
                profiles,
                model_name=args.model,
                output_dir=results_dir,
            )
            logger.info("Plots written to %s", results_dir)
        else:
            logger.warning("plot_benchmark.py not found -- skipping plots")
    except Exception:
        logger.exception("Plot generation failed -- skipping")

stdout_console.print(f"\n[bold]Results dir:[/] {results_dir}")
