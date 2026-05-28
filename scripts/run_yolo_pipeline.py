"""Runs the YOLO → LLM baseline pipeline on an image.

Moved from ``src/moment_to_action/edgeperceive/pipeline/run_yolo_pipeline.py``.

Usage:
    uv run python scripts/run_yolo_pipeline.py --image weapon.jpg
    uv run python scripts/run_yolo_pipeline.py --image weapon.jpg --device npu
"""

from __future__ import annotations

import argparse
import logging

from rich.console import Console
from rich.logging import RichHandler

from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.models import ModelID, ModelManager
from moment_to_action.paths import PathManager

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True, console=Console(stderr=True)),
    ],
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
args = parser.parse_args()

device = ComputeUnit.NPU if args.device == "npu" else ComputeUnit.CPU
compute_backend = ComputeBackend(preferred_unit=device)
manager = ModelManager(PathManager())

# ── load model ─────────────────────────────────────────────────────────────
model = manager.get_model(ModelID.YOLO_V8)
model.load(compute_backend)

logger.info("Model loaded. Pipeline wiring (ImageDetectionStage) is deferred to PR 2.")
model.unload()
