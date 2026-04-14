"""Profile a single model across CPU, GPU, and NPU compute units.

For each requested compute unit, loads the model, runs warmup iterations,
then collects latency percentiles, peak memory, power draw, model size,
and accuracy against a CPU oracle (using default test images).

Results are saved to a timestamped sub-directory under ``scripts/`` and
plotted as bar charts.

Usage::

    uv run python scripts/benchmark_model.py --model yolo --units cpu gpu npu
    uv run python scripts/benchmark_model.py --model mobileclip --units cpu gpu npu
    uv run python scripts/benchmark_model.py --model smolvlm2 --units cpu --n-runs 10
    uv run python scripts/benchmark_model.py --model yolo --eval-images custom/*.jpg
    uv run python scripts/benchmark_model.py --model yolo --results-dir scripts/my_run
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

# Emit a Python + native C stack trace to stderr on any fatal signal
# (SIGSEGV, SIGBUS, SIGFPE, SIGABRT) that escapes the subprocess probe.
faulthandler.enable()

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

_UNIT_MAP: dict[str, ComputeUnit] = {
    "cpu": ComputeUnit.CPU,
    "gpu": ComputeUnit.GPU,
    "npu": ComputeUnit.NPU,
}

parser = argparse.ArgumentParser(
    description="Profile a model across compute units and report latency, memory, accuracy, and cost metrics."
)
parser.add_argument(
    "--model",
    required=True,
    choices=["yolo", "mobileclip", "smolvlm2", "qwen3"],
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

# Default eval images: all JPG files in tests/int/images/
_DEFAULT_EVAL_IMAGES = sorted(
    (Path(__file__).parent.parent / "tests" / "int" / "images").glob("*.jpg")
)

parser.add_argument(
    "--eval-images",
    nargs="+",
    type=Path,
    default=_DEFAULT_EVAL_IMAGES,
    metavar="IMAGE",
    help=(
        "Image paths used for accuracy evaluation (CPU oracle vs variant). "
        f"Defaults to tests/int/images/*.jpg ({len(_DEFAULT_EVAL_IMAGES)} images). "
        "Pass empty list to skip accuracy evaluation."
    ),
)
parser.add_argument(
    "--results-dir",
    type=Path,
    default=None,
    metavar="DIR",
    help=(
        "Directory to write JSON, CSV, and plot PNGs. "
        "Defaults to scripts/tmp_results_<timestamp>/"
    ),
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    metavar="FILE",
    help="Additional path to write JSON results (legacy flag — results-dir is preferred).",
)
parser.add_argument(
    "--save-registry",
    action="store_true",
    help="Persist results into the default VariantRegistry cache.",
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

# ── Resolve results directory ──────────────────────────────────────────────────

_SCRIPTS_DIR = Path(__file__).parent
if args.results_dir is not None:
    results_dir: Path = args.results_dir
else:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    results_dir = _SCRIPTS_DIR / f"tmp_results__{args.model}_{timestamp}"
results_dir.mkdir(parents=True, exist_ok=True)
logger.info("Results will be saved to %s", results_dir)

# ── Setup ──────────────────────────────────────────────────────────────────────

eval_image_paths: list[Path] = args.eval_images or []

# Build model benchmark objects with eval images wired in.
_MODEL_CHOICES: dict[str, tuple[ModelID, ModelBenchmark]] = {
    "yolo": (ModelID.YOLO_V8, YOLOBenchmark(eval_image_paths=eval_image_paths)),
    "mobileclip": (ModelID.MOBILECLIP_S2, MobileCLIPBenchmark(eval_image_paths=eval_image_paths)),
    "smolvlm2": (ModelID.SMOLVLM2_2_2B, SmolVLM2Benchmark()),
    "qwen3": (ModelID.QWEN2_5_4B, Qwen3Benchmark()),
}

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
table.add_column("Accuracy", justify="right")
table.add_column("Peak Mem (MB)", justify="right")
table.add_column("Max Batch", justify="right")
table.add_column("Power (mW)", justify="right")
table.add_column("Energy/inf (mJ)", justify="right")
table.add_column("Model Size (MB)", justify="right")

for p in profiles:
    unit_str = p.variant_id.compute_unit.value
    power = f"{p.cost.power_mw:.0f}" if p.cost.power_mw is not None else "—"
    energy = (
        f"{p.cost.energy_per_inference_mj:.3f}"
        if p.cost.energy_per_inference_mj is not None
        else "—"
    )
    accuracy = f"{p.accuracy:.3f}" if p.accuracy is not None else "—"
    table.add_row(
        unit_str,
        f"{p.load_latency_ms:.1f}",
        f"{p.inference_mean_ms:.2f}",
        f"{p.inference_p50_ms:.2f}",
        f"{p.inference_p95_ms:.2f}",
        f"{p.inference_p99_ms:.2f}",
        accuracy,
        f"{p.peak_memory_mb:.0f}",
        str(p.max_batch_size),
        power,
        energy,
        f"{p.model_size_bytes / (1024**2):.1f}",
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

# ── Save JSON to results dir ───────────────────────────────────────────────────

payload = {
    "model": args.model,
    "model_id": model_id.value,
    "config": {
        "n_warmup": config.n_warmup,
        "n_runs": config.n_runs,
    },
    "eval_images": [str(p) for p in eval_image_paths],
    "profiles": [p.json() for p in profiles],
}
json_path = results_dir / f"{args.model}_profiles.json"
json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
logger.info("JSON results written to %s", json_path)

if args.output is not None:
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("JSON also written to %s", args.output)

# ── Save CSV to results dir ────────────────────────────────────────────────────

csv_path = results_dir / f"{args.model}_profiles.csv"
_CSV_FIELDS = [
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
with csv_path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    for p in profiles:
        writer.writerow({
            "model": args.model,
            "model_id": p.variant_id.model_id.value,
            "compute_unit": p.variant_id.compute_unit.value,
            "hardware_target": p.hardware_target,
            "load_latency_ms": p.load_latency_ms,
            "inference_mean_ms": p.inference_mean_ms,
            "inference_p50_ms": p.inference_p50_ms,
            "inference_p95_ms": p.inference_p95_ms,
            "inference_p99_ms": p.inference_p99_ms,
            "accuracy": p.accuracy if p.accuracy is not None else "",
            "peak_memory_mb": p.peak_memory_mb,
            "max_batch_size": p.max_batch_size,
            "power_mw": p.cost.power_mw if p.cost.power_mw is not None else "",
            "energy_per_inference_mj": (
                p.cost.energy_per_inference_mj
                if p.cost.energy_per_inference_mj is not None
                else ""
            ),
            "model_size_bytes": p.model_size_bytes,
            "n_runs": p.n_runs,
            "profiled_at": p.profiled_at.isoformat(),
        })
logger.info("CSV table written to %s", csv_path)

# ── Generate plots ─────────────────────────────────────────────────────────────

if not args.no_plot:
    try:
        from plot_benchmark import plot_profiles  # noqa: PLC0415
        plot_profiles(profiles, model_name=args.model, output_dir=results_dir)
        logger.info("Plots written to %s", results_dir)
    except ImportError:
        # plot_benchmark lives next to this script — add scripts/ to path
        import importlib.util  # noqa: PLC0415
        spec = importlib.util.spec_from_file_location(
            "plot_benchmark", _SCRIPTS_DIR / "plot_benchmark.py"
        )
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            mod.plot_profiles(profiles, model_name=args.model, output_dir=results_dir)  # type: ignore[attr-defined]
            logger.info("Plots written to %s", results_dir)
        else:
            logger.warning("plot_benchmark.py not found — skipping plots")
    except Exception:  # noqa: BLE001
        logger.exception("Plot generation failed — skipping")

rich.print(f"\n[bold]Results dir:[/] {results_dir}")
