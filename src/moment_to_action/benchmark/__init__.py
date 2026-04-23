"""Benchmarking APIs for profiling model variants across compute backends."""

from __future__ import annotations

from ._benchmarks import (
    MobileCLIPBenchmark,
    ModelBenchmark,
    SigLIPBenchmark,
    YOLOBenchmark,
)
from ._datasets import (
    BaseDataset,
    CocoDataset,
)
from ._detection_metrics import DetectionMetrics, compute_detection_map
from ._harness import BenchmarkHarness
from ._retrieval_metrics import RetrievalMetrics, compute_retrieval_metrics
from ._types import BenchmarkConfig, CostProfile, VariantID, VariantProfile
from ._variant_registry import VariantRegistry

__all__ = [
    "BaseDataset",
    "BenchmarkConfig",
    "BenchmarkHarness",
    "CocoDataset",
    "CostProfile",
    "DetectionMetrics",
    "MobileCLIPBenchmark",
    "ModelBenchmark",
    "RetrievalMetrics",
    "SigLIPBenchmark",
    "VariantID",
    "VariantProfile",
    "VariantRegistry",
    "YOLOBenchmark",
    "compute_detection_map",
    "compute_retrieval_metrics",
]
