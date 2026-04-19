from __future__ import annotations

from ._base import ModelBenchmark
from ._grounding_dino import GroundingDINOBenchmark
from ._mobileclip import MobileCLIPBenchmark
from ._qwen3 import Qwen3Benchmark
from ._siglip import SigLIPBenchmark
from ._smolvlm2 import SmolVLM2Benchmark
from ._whisper import WhisperTinyBenchmark
from ._yolo import YOLOBenchmark

__all__ = [
    "GroundingDINOBenchmark",
    "MobileCLIPBenchmark",
    "ModelBenchmark",
    "Qwen3Benchmark",
    "SigLIPBenchmark",
    "SmolVLM2Benchmark",
    "WhisperTinyBenchmark",
    "YOLOBenchmark",
]
