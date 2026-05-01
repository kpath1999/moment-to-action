"""Run COCO detection and retrieval benchmarking across compute units."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from moment_to_action.benchmark import (
    BenchmarkConfig,
    CocoDataset,
    MobileCLIPBenchmark,
    RFDETRBenchmark,
    SigLIPBenchmark,
    SSDMobileNetV2Benchmark,
    YOLOBenchmark,
)
from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.models import ModelManager
from moment_to_action.models._variants.yolov12 import get_yolov12_model_for_unit

logger = logging.getLogger(__name__)

_UNIT_MAP: dict[str, ComputeUnit] = {
    "cpu": ComputeUnit.CPU,
    "gpu": ComputeUnit.GPU,
    "npu": ComputeUnit.NPU,
}

_UNSUPPORTED_MODELS = {
    "tinyclip_8m": "runner not implemented yet",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run COCO detection/retrieval benchmarking.")
    parser.add_argument(
        "--n-images",
        type=int,
        default=500,
        help="Number of COCO val images to use.",
    )
    parser.add_argument(
        "--model",
        choices=[
            "yolo_v12_n",
            "rf_detr_n",
            "ssd_mobilenetv2",
            "tinyclip_8m",
            "mobileclip_s2",
            "siglip",
            "all",
        ],
        default="all",
        help="Which model(s) to evaluate.",
    )
    parser.add_argument(
        "--edge-unit",
        choices=["cpu", "gpu", "npu"],
        default="npu",
        help="Compute unit used for edge model evaluation.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="YOLO confidence threshold used for prediction decoding.",
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


def _run_yolo_eval(
    dataset: CocoDataset,
    manager: ModelManager,
    unit: ComputeUnit,
    conf_threshold: float,
) -> dict[str, float | None]:
    logger.info("YOLO eval config: unit=%s conf_threshold=%.6f", unit.value, conf_threshold)
    backend = ComputeBackend(preferred_unit=unit)
    # Resolve the correct YOLOv12 model path for the compute unit
    model_path = get_yolov12_model_for_unit(unit=unit)
    benchmark = YOLOBenchmark(
        coco_dataset=dataset,
        conf_threshold=conf_threshold,
        model_path=str(model_path),
    )
    profile = benchmark.profile(
        backend=backend,
        manager=manager,
        config=BenchmarkConfig(n_warmup=3, n_runs=10, batch_sizes=[1]),
    )

    payload: dict[str, float | None] = {"inference_mean_ms": profile.inference_mean_ms}
    if profile.accuracy_details is not None:
        payload.update(profile.accuracy_details)
    return payload


def _run_mobileclip_eval(
    dataset: CocoDataset,
    manager: ModelManager,
    unit: ComputeUnit,
) -> dict[str, float | None]:
    backend = ComputeBackend(preferred_unit=unit)
    benchmark = MobileCLIPBenchmark(coco_dataset=dataset)
    profile = benchmark.profile(
        backend=backend,
        manager=manager,
        config=BenchmarkConfig(n_warmup=3, n_runs=10, batch_sizes=[1]),
    )

    payload: dict[str, float | None] = {"inference_mean_ms": profile.inference_mean_ms}
    if profile.accuracy_details is not None:
        payload.update(profile.accuracy_details)
    return payload


def _run_siglip_eval(
    dataset: CocoDataset,
    manager: ModelManager,
    unit: ComputeUnit,
) -> dict[str, float | None]:
    backend = ComputeBackend(preferred_unit=unit)
    benchmark = SigLIPBenchmark(coco_dataset=dataset)
    profile = benchmark.profile(
        backend=backend,
        manager=manager,
        config=BenchmarkConfig(n_warmup=3, n_runs=10, batch_sizes=[1]),
    )

    payload: dict[str, float | None] = {"inference_mean_ms": profile.inference_mean_ms}
    if profile.accuracy_details is not None:
        payload.update(profile.accuracy_details)
    return payload


def _run_ssd_eval(
    dataset: CocoDataset,
    manager: ModelManager,
    unit: ComputeUnit,
) -> dict[str, float | None]:
    backend = ComputeBackend(preferred_unit=unit)
    benchmark = SSDMobileNetV2Benchmark(coco_dataset=dataset)
    profile = benchmark.profile(
        backend=backend,
        manager=manager,
        config=BenchmarkConfig(n_warmup=3, n_runs=10, batch_sizes=[1]),
    )

    payload: dict[str, float | None] = {"inference_mean_ms": profile.inference_mean_ms}
    if profile.accuracy_details is not None:
        payload.update(profile.accuracy_details)
    return payload


def _run_rfdetr_eval(
    dataset: CocoDataset,
    manager: ModelManager,
    unit: ComputeUnit,
) -> dict[str, float | None]:
    backend = ComputeBackend(preferred_unit=unit)
    benchmark = RFDETRBenchmark(coco_dataset=dataset)
    profile = benchmark.profile(
        backend=backend,
        manager=manager,
        config=BenchmarkConfig(n_warmup=3, n_runs=10, batch_sizes=[1]),
    )

    payload: dict[str, float | None] = {"inference_mean_ms": profile.inference_mean_ms}
    if profile.accuracy_details is not None:
        payload.update(profile.accuracy_details)
    return payload


def main() -> None:
    """Execute COCO model evaluation flow."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    dataset = CocoDataset(n_images=args.n_images)
    manager = ModelManager()
    results: dict[str, object] = {
        "dataset": dataset.dataset_name,
        "n_images": len(dataset.images()),
        "compute_unit": _UNIT_MAP[args.edge_unit].value,
    }

    if args.model in {"yolo_v12_n", "all"}:
        logger.info("Running YOLOv12-n COCO detection evaluation...")
        results["yolo_v12_n"] = _run_yolo_eval(
            dataset=dataset,
            manager=manager,
            unit=_UNIT_MAP[args.edge_unit],
            conf_threshold=args.conf_threshold,
        )

    if args.model in {"ssd_mobilenetv2", "all"}:
        logger.info("Running SSD-MobileNet-v2 COCO detection evaluation...")
        results["ssd_mobilenetv2"] = _run_ssd_eval(
            dataset=dataset,
            manager=manager,
            unit=_UNIT_MAP[args.edge_unit],
        )

    if args.model in {"rf_detr_n", "all"}:
        logger.info("Running RF-DETR-n COCO detection evaluation...")
        results["rf_detr_n"] = _run_rfdetr_eval(
            dataset=dataset,
            manager=manager,
            unit=_UNIT_MAP[args.edge_unit],
        )

    if args.model in {"mobileclip_s2", "all"}:
        logger.info("Running MobileCLIP-S2 COCO retrieval evaluation...")
        results["mobileclip_s2"] = _run_mobileclip_eval(
            dataset=dataset,
            manager=manager,
            unit=_UNIT_MAP[args.edge_unit],
        )

    if args.model in {"siglip", "all"}:
        logger.info("Running SigLIP COCO retrieval evaluation...")
        results["siglip"] = _run_siglip_eval(
            dataset=dataset,
            manager=manager,
            unit=_UNIT_MAP[args.edge_unit],
        )

    for model_name, reason in _UNSUPPORTED_MODELS.items():
        if args.model in {model_name, "all"}:
            results[model_name] = {
                "status": "unsupported",
                "reason": reason,
            }

    print(json.dumps(results, indent=2))

    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Saved COCO evaluation results to %s", args.output)


if __name__ == "__main__":
    main()
