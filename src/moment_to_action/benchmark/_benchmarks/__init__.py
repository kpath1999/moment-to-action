from __future__ import annotations

from ._base import ModelBenchmark
from ._mobileclip import MobileCLIPBenchmark
from ._rf_detr_n import RFDETRBenchmark
from ._siglip import SigLIPBenchmark
from ._ssd_mobilenetv2 import SSDMobileNetV2Benchmark
from ._yolo import YOLOBenchmark

__all__ = [
    "MobileCLIPBenchmark",
    "ModelBenchmark",
    "RFDETRBenchmark",
    "SSDMobileNetV2Benchmark",
    "SigLIPBenchmark",
    "YOLOBenchmark",
]
