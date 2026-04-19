"""Benchmarking APIs for profiling model variants across compute backends."""

from __future__ import annotations

from ._benchmarks import (
    GroundingDINOBenchmark,
    MobileCLIPBenchmark,
    ModelBenchmark,
    Qwen3Benchmark,
    SigLIPBenchmark,
    SmolVLM2Benchmark,
    WhisperTinyBenchmark,
    YOLOBenchmark,
)
from ._datasets import (
    BaseDataset,
    CocoDataset,
    GSM8KDataset,
    GSM8KItem,
    LibriSpeechDataset,
    LibriSpeechItem,
    MsrvttDataset,
    MsrvttItem,
)
from ._detection_metrics import DetectionMetrics, compute_detection_map
from ._harness import BenchmarkHarness
from ._oracle_ground_truth import (
    OracleBox,
    OracleClassification,
    OracleDetection,
    OracleGroundTruth,
    OracleStore,
)
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
    "GSM8KDataset",
    "GSM8KItem",
    "GroundingDINOBenchmark",
    "LibriSpeechDataset",
    "LibriSpeechItem",
    "MobileCLIPBenchmark",
    "ModelBenchmark",
    "MsrvttDataset",
    "MsrvttItem",
    "OracleBox",
    "OracleClassification",
    "OracleDetection",
    "OracleGroundTruth",
    "OracleStore",
    "Qwen3Benchmark",
    "RetrievalMetrics",
    "SigLIPBenchmark",
    "SmolVLM2Benchmark",
    "VariantID",
    "VariantProfile",
    "VariantRegistry",
    "WhisperTinyBenchmark",
    "YOLOBenchmark",
    "compute_detection_map",
    "compute_retrieval_metrics",
]
