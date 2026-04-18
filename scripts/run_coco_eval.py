"""Run COCO pseudo-ground-truth evaluation for YOLO and MobileCLIP."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from moment_to_action.benchmark import (
    BenchmarkConfig,
    CocoDataset,
    GroundingDINOBenchmark,
    MobileCLIPBenchmark,
    OracleStore,
    SigLIPBenchmark,
    YOLOBenchmark,
)
from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.models import ModelManager

logger = logging.getLogger(__name__)

_UNIT_MAP: dict[str, ComputeUnit] = {
    "cpu": ComputeUnit.CPU,
    "gpu": ComputeUnit.GPU,
    "npu": ComputeUnit.NPU,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run COCO pseudo-ground-truth evaluation for YOLO and MobileCLIP."
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=500,
        help="Number of COCO val images to use.",
    )
    parser.add_argument(
        "--model",
        choices=["yolo", "mobileclip", "both", "oracle"],
        default="both",
        help="Which edge model(s) to evaluate. Use 'oracle' to only generate pseudo-GT labels.",
    )
    parser.add_argument(
        "--oracle-unit",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Compute unit used for GroundingDINO/SigLIP oracle passes.",
    )
    parser.add_argument(
        "--edge-unit",
        choices=["cpu", "gpu", "npu"],
        default="npu",
        help="Compute unit used for edge model evaluation.",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Reuse existing COCO oracle records and skip GroundingDINO/SigLIP runs.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="YOLO confidence threshold used for prediction decoding.",
    )
    parser.add_argument(
        "--oracle-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "oracle",
        help=(
            "Directory where oracle pseudo-GT JSON files are read and written. "
            "Defaults to data/oracle/ in the repo root so the files can be committed "
            "and transferred to the edge device."
        ),
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


def _run_oracle_passes(
    dataset: CocoDataset,
    manager: ModelManager,
    unit: ComputeUnit,
    oracle_store: OracleStore,
) -> None:
    backend = ComputeBackend(preferred_unit=unit)
    config = BenchmarkConfig(n_warmup=1, n_runs=1, batch_sizes=[1])

    grounding = GroundingDINOBenchmark(coco_dataset=dataset, oracle_store=oracle_store)
    grounding.profile(backend=backend, manager=manager, config=config)

    siglip = SigLIPBenchmark(coco_dataset=dataset, oracle_store=oracle_store)
    siglip.profile(backend=backend, manager=manager, config=config)


def _run_yolo_eval(
    dataset: CocoDataset,
    manager: ModelManager,
    unit: ComputeUnit,
    conf_threshold: float,
) -> dict[str, float | None]:
    backend = ComputeBackend(preferred_unit=unit)
    benchmark = YOLOBenchmark(coco_dataset=dataset, conf_threshold=conf_threshold)
    profile = benchmark.profile(
        backend=backend,
        manager=manager,
        config=BenchmarkConfig(n_warmup=3, n_runs=10, batch_sizes=[1]),
    )

    payload: dict[str, float | None] = {
        "accuracy": profile.accuracy,
        "inference_mean_ms": profile.inference_mean_ms,
    }
    if profile.accuracy_details is not None:
        payload.update(profile.accuracy_details)
    return payload


def _run_mobileclip_eval(
    dataset: CocoDataset,
    manager: ModelManager,
    unit: ComputeUnit,
    oracle_store: OracleStore,
) -> dict[str, float | None]:
    backend = ComputeBackend(preferred_unit=unit)
    benchmark = MobileCLIPBenchmark(coco_dataset=dataset, oracle_store=oracle_store)
    profile = benchmark.profile(
        backend=backend,
        manager=manager,
        config=BenchmarkConfig(n_warmup=3, n_runs=10, batch_sizes=[1]),
    )

    payload: dict[str, float | None] = {
        "accuracy": profile.accuracy,
        "inference_mean_ms": profile.inference_mean_ms,
    }
    if profile.accuracy_details is not None:
        payload.update(profile.accuracy_details)
    return payload


def main() -> None:
    """Execute the COCO pseudo-oracle generation and edge-model evaluation flow."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    dataset = CocoDataset(n_images=args.n_images)
    manager = ModelManager()
    oracle_dir: Path = args.oracle_dir
    oracle_store = OracleStore(path=oracle_dir / f"oracle_{dataset.dataset_name}.json")
    needs_oracle = args.model in {"mobileclip", "both", "oracle"}

    if not needs_oracle:
        logger.info("Skipping oracle generation for YOLO-only run (native COCO GT is used).")
    elif not args.skip_oracle and oracle_store.load() is None:
        logger.info("Generating COCO oracle pseudo-ground-truth with GroundingDINO and SigLIP...")
        _run_oracle_passes(dataset, manager, _UNIT_MAP[args.oracle_unit], oracle_store)
    elif not args.skip_oracle:
        logger.info("Refreshing COCO oracle pseudo-ground-truth...")
        _run_oracle_passes(dataset, manager, _UNIT_MAP[args.oracle_unit], oracle_store)
    else:
        logger.info("Skipping oracle generation and reusing cached COCO oracle data.")

    results: dict[str, object] = {
        "dataset": dataset.dataset_name,
        "n_images": len(dataset.images()),
        "oracle_path": str(oracle_store.path),
    }

    if args.model in {"yolo", "both"}:
        logger.info("Running YOLO COCO native-GT evaluation...")
        results["yolo"] = _run_yolo_eval(
            dataset=dataset,
            manager=manager,
            unit=_UNIT_MAP[args.edge_unit],
            conf_threshold=args.conf_threshold,
        )

    if args.model in {"mobileclip", "both"}:
        logger.info("Running MobileCLIP COCO pseudo-GT evaluation...")
        results["mobileclip"] = _run_mobileclip_eval(
            dataset=dataset,
            manager=manager,
            unit=_UNIT_MAP[args.edge_unit],
            oracle_store=oracle_store,
        )

    print(json.dumps(results, indent=2))

    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Saved COCO evaluation results to %s", args.output)


if __name__ == "__main__":
    main()
