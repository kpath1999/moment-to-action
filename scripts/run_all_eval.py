"""Run all benchmark task families from one entry point."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from moment_to_action.benchmark import (
    BenchmarkConfig,
    CocoDataset,
    GroundingDINOBenchmark,
    GSM8KDataset,
    LibriSpeechDataset,
    MobileCLIPBenchmark,
    MsrvttDataset,
    OracleStore,
    Qwen3Benchmark,
    SigLIPBenchmark,
    SmolVLM2Benchmark,
    WhisperTinyBenchmark,
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

_TASK_CHOICES = ["yolo", "mobileclip", "smolvlm2", "qwen3", "whisper", "all"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all benchmark task families.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=_TASK_CHOICES,
        default=["all"],
        help="Task families to run.",
    )
    parser.add_argument("--n-images", type=int, default=500, help="Number of COCO images.")
    parser.add_argument(
        "--n-items",
        type=int,
        default=500,
        help="Number of items for non-COCO datasets.",
    )
    parser.add_argument(
        "--msrvtt-local-dir",
        type=Path,
        default=None,
        help=(
            "Path to a local MSRVTT-QA directory containing '{split}_qa.json' "
            "and a 'videos/' sub-folder. When set, skips the HuggingFace download."
        ),
    )
    parser.add_argument(
        "--msrvtt-dataset-id",
        default="lmms-lab/MSRVTT-QA",
        help=(
            "Hugging Face dataset id for SmolVLM2 evaluation. "
            "If private, set HF_TOKEN/HUGGING_FACE_HUB_TOKEN."
        ),
    )
    parser.add_argument(
        "--msrvtt-split",
        default="test",
        help="Dataset split for SmolVLM2 evaluation (e.g. test, validation, train).",
    )
    parser.add_argument(
        "--smolvlm2-n-items",
        type=int,
        default=100,
        help="Number of MSRVTT-QA clips to evaluate for SmolVLM2 (default: 100).",
    )
    parser.add_argument(
        "--mobileclip-oracle-unit",
        "--oracle-unit",
        dest="mobileclip_oracle_unit",
        choices=["cpu", "gpu"],
        default="cpu",
        help=(
            "Compute unit used only when generating MobileCLIP pseudo-ground-truth "
            "labels (GroundingDINO + SigLIP)."
        ),
    )
    parser.add_argument(
        "--edge-unit",
        choices=["cpu", "gpu", "npu"],
        default="cpu",
        help="Compute unit used for edge model evaluation.",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Skip MobileCLIP oracle generation.",
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
        help="Directory used to persist oracle pseudo-GT files.",
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


def _task_enabled(task_name: str, selected: set[str]) -> bool:
    return "all" in selected or task_name in selected


def main() -> None:  # noqa: C901
    """Execute selected benchmark task families from one CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    selected = set(args.tasks)

    manager = ModelManager()
    edge_backend = ComputeBackend(preferred_unit=_UNIT_MAP[args.edge_unit])

    results: dict[str, object] = {
        "edge_unit": edge_backend.active_unit.value,
        "tasks": sorted(selected),
    }

    if _task_enabled("yolo", selected) or _task_enabled("mobileclip", selected):
        coco_dataset = CocoDataset(n_images=args.n_images)
        oracle_dir: Path = args.oracle_dir
        oracle_store = OracleStore(path=oracle_dir / f"oracle_{coco_dataset.dataset_name}.json")

        if _task_enabled("mobileclip", selected) and not args.skip_oracle:
            logger.info("Refreshing COCO oracle pseudo-ground-truth for MobileCLIP...")
            oracle_backend = ComputeBackend(preferred_unit=_UNIT_MAP[args.mobileclip_oracle_unit])
            config = BenchmarkConfig(n_warmup=1, n_runs=1, batch_sizes=[1])
            GroundingDINOBenchmark(
                coco_dataset=coco_dataset,
                oracle_store=oracle_store,
            ).profile(
                backend=oracle_backend,
                manager=manager,
                config=config,
            )
            SigLIPBenchmark(coco_dataset=coco_dataset, oracle_store=oracle_store).profile(
                backend=oracle_backend,
                manager=manager,
                config=config,
            )

        results["coco"] = {
            "dataset": coco_dataset.dataset_name,
            "n_images": len(coco_dataset.images()),
            "oracle_path": str(oracle_store.path),
        }

        if _task_enabled("yolo", selected):
            profile = YOLOBenchmark(
                coco_dataset=coco_dataset,
                conf_threshold=args.conf_threshold,
            ).profile(
                backend=edge_backend,
                manager=manager,
                config=BenchmarkConfig(n_warmup=3, n_runs=10, batch_sizes=[1]),
            )
            results["yolo"] = {
                "accuracy": profile.accuracy,
                "inference_mean_ms": profile.inference_mean_ms,
                "accuracy_details": profile.accuracy_details,
            }

        if _task_enabled("mobileclip", selected):
            profile = MobileCLIPBenchmark(
                coco_dataset=coco_dataset,
                oracle_store=oracle_store,
            ).profile(
                backend=edge_backend,
                manager=manager,
                config=BenchmarkConfig(n_warmup=3, n_runs=10, batch_sizes=[1]),
            )
            results["mobileclip"] = {
                "accuracy": profile.accuracy,
                "inference_mean_ms": profile.inference_mean_ms,
                "accuracy_details": profile.accuracy_details,
            }

    if _task_enabled("smolvlm2", selected):
        try:
            msrvtt_dataset = MsrvttDataset(
                n_items=args.smolvlm2_n_items,
                dataset_id=args.msrvtt_dataset_id,
                split=args.msrvtt_split,
                local_dir=args.msrvtt_local_dir,
            )
        except RuntimeError as exc:
            logger.error("Skipping smolvlm2 benchmark: %s", exc)  # noqa: TRY400
            results["smolvlm2"] = {
                "error": str(exc),
            }
        else:
            profile = SmolVLM2Benchmark(msrvtt_dataset=msrvtt_dataset).profile(
                backend=edge_backend,
                manager=manager,
                config=BenchmarkConfig(n_warmup=1, n_runs=5, batch_sizes=[1]),
            )
            results["smolvlm2"] = {
                "dataset": msrvtt_dataset.dataset_name,
                "n_items": len(msrvtt_dataset.items()),
                "accuracy": profile.accuracy,
                "inference_mean_ms": profile.inference_mean_ms,
                "accuracy_details": profile.accuracy_details,
            }

    if _task_enabled("qwen3", selected):
        gsm8k_dataset = GSM8KDataset(n_items=args.n_items)
        profile = Qwen3Benchmark(gsm8k_dataset=gsm8k_dataset).profile(
            backend=edge_backend,
            manager=manager,
            config=BenchmarkConfig(n_warmup=1, n_runs=5, batch_sizes=[1]),
        )
        results["qwen3"] = {
            "dataset": gsm8k_dataset.dataset_name,
            "n_items": len(gsm8k_dataset.items()),
            "accuracy": profile.accuracy,
            "inference_mean_ms": profile.inference_mean_ms,
            "accuracy_details": profile.accuracy_details,
        }

    if _task_enabled("whisper", selected):
        librispeech_dataset = LibriSpeechDataset(n_items=args.n_items)
        profile = WhisperTinyBenchmark(librispeech_dataset=librispeech_dataset).profile(
            backend=edge_backend,
            manager=manager,
            config=BenchmarkConfig(n_warmup=1, n_runs=5, batch_sizes=[1]),
        )
        results["whisper_tiny"] = {
            "dataset": librispeech_dataset.dataset_name,
            "n_items": len(librispeech_dataset.items()),
            "accuracy": profile.accuracy,
            "inference_mean_ms": profile.inference_mean_ms,
            "accuracy_details": profile.accuracy_details,
        }

    print(json.dumps(results, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Saved unified evaluation results to %s", args.output)


if __name__ == "__main__":
    main()
