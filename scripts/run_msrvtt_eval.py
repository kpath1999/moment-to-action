"""Run MSRVTT-QA benchmark evaluation for SmolVLM2."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from moment_to_action.benchmark import BenchmarkConfig, MsrvttDataset, SmolVLM2Benchmark
from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.models import ModelManager

logger = logging.getLogger(__name__)

_UNIT_MAP: dict[str, ComputeUnit] = {
    "cpu": ComputeUnit.CPU,
    "gpu": ComputeUnit.GPU,
    "npu": ComputeUnit.NPU,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MSRVTT-QA benchmark evaluation for SmolVLM2.")
    parser.add_argument(
        "--n-items",
        type=int,
        default=500,
        help="Number of MSRVTT-QA items to use.",
    )
    parser.add_argument(
        "--edge-unit",
        choices=["cpu", "gpu", "npu"],
        default="cpu",
        help="Compute unit used for model evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for JSON results.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    """Execute SmolVLM2 exact-match evaluation on MSRVTT-QA."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    dataset = MsrvttDataset(n_items=args.n_items)
    manager = ModelManager()
    backend = ComputeBackend(preferred_unit=_UNIT_MAP[args.edge_unit])

    benchmark = SmolVLM2Benchmark(msrvtt_dataset=dataset)
    profile = benchmark.profile(
        backend=backend,
        manager=manager,
        config=BenchmarkConfig(n_warmup=1, n_runs=5, batch_sizes=[1]),
    )

    payload: dict[str, object] = {
        "dataset": dataset.dataset_name,
        "n_items": len(dataset.items()),
        "smolvlm2": {
            "accuracy": profile.accuracy,
            "inference_mean_ms": profile.inference_mean_ms,
            "accuracy_details": profile.accuracy_details,
        },
    }

    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved MSRVTT evaluation results to %s", args.output)


if __name__ == "__main__":
    main()
