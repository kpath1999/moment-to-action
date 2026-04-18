"""Benchmarking APIs for profiling model variants across compute backends."""

from __future__ import annotations

from ._base import ModelBenchmark
from ._coco_dataset import CocoDataset
from ._detection_metrics import DetectionMetrics, compute_detection_map
from ._grounding_dino import GroundingDINOBenchmark
from ._gsm8k_dataset import GSM8KDataset, GSM8KItem
from ._harness import BenchmarkHarness
from ._librispeech_dataset import LibriSpeechDataset, LibriSpeechItem
from ._mobileclip import MobileCLIPBenchmark
from ._msrvtt_dataset import MsrvttDataset, MsrvttItem
from ._oracle_ground_truth import (
    OracleBox,
    OracleClassification,
    OracleDetection,
    OracleGroundTruth,
    OracleStore,
)
from ._qwen3 import Qwen3Benchmark
from ._retrieval_metrics import RetrievalMetrics, compute_retrieval_metrics
from ._siglip import SigLIPBenchmark
from ._smolvlm2 import SmolVLM2Benchmark
from ._types import BenchmarkConfig, CostProfile, VariantID, VariantProfile
from ._variant_registry import VariantRegistry
from ._whisper import WhisperTinyBenchmark
from ._yolo import YOLOBenchmark

__all__ = [
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
