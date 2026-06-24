"""Runs the YOLO object-detection pipeline on a single image.

Moved from ``src/moment_to_action/edgeperceive/pipeline/run_yolo_pipeline.py``.

Usage:
    uv run python scripts/run_yolo_pipeline.py --image weapon.jpg
    uv run python scripts/run_yolo_pipeline.py --image weapon.jpg --device npu --conf 0.4
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2
from rich.console import Console
from rich.logging import RichHandler

from moment_to_action.config import load_config
from moment_to_action.hardware import ComputeUnit, Platform
from moment_to_action.messages import DetectionMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.models import ModelID, ModelManager
from moment_to_action.models.image.detection._base import ImageDetectionModel
from moment_to_action.paths import PathManager
from moment_to_action.stages import Pipeline
from moment_to_action.stages.image import ImageDetectionStage

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

frame = cv2.imread(args.image)
if frame is None:
    logger.error("Could not load image: %s", args.image)
    raise SystemExit(1)

device = ComputeUnit.NPU if args.device == "npu" else ComputeUnit.CPU
path_manager = PathManager()
config = load_config(path_manager.app_config_file)
platform = Platform(config)
manager = ModelManager(path_manager)

# ── load model ─────────────────────────────────────────────────────────────
model = manager.get_model(ModelID.YOLO_V8, confidence_threshold=args.conf)
if not isinstance(model, ImageDetectionModel):
    err_msg = f"Expected ImageDetectionModel, got {type(model).__name__}"
    raise TypeError(err_msg)
model.load(platform, device)

# ── build and run pipeline ─────────────────────────────────────────────────
stage = ImageDetectionStage(model=model)
pipeline = Pipeline(stages=[stage])

raw_msg = RawFrameMessage(
    frame=frame,
    timestamp=time.time(),
    width=frame.shape[1],
    height=frame.shape[0],
)
result = pipeline.run(raw_msg)

# ── display results ────────────────────────────────────────────────────────
if isinstance(result, DetectionMessage):
    logger.info("Found %d detection(s):", len(result.detections))
    for det in result.detections:
        logger.info("  %s  conf=%.2f  bbox=%s", det.label, det.confidence, det.bbox)
else:
    logger.warning("Pipeline returned no detection result.")

model.unload()
