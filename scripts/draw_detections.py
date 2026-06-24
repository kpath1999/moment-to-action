"""Runs YOLO detection on an image and saves an annotated copy with bounding boxes.

Usage:
    uv run python scripts/draw_detections.py --image pedestrian.jpg
    uv run python scripts/draw_detections.py --image pedestrian.jpg --out result.jpg
    uv run python scripts/draw_detections.py --image pedestrian.jpg --conf 0.4 --device npu
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
from rich.console import Console
from rich.logging import RichHandler

from moment_to_action.config import load_config
from moment_to_action.hardware import ComputeUnit, Platform
from moment_to_action.models import ModelID, ModelManager, YOLOModel
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
parser.add_argument("--out", default=None, help="Output path (default: annotated_<input>)")
parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
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
if not isinstance(model, YOLOModel):
    err_msg = f"Expected YOLOModel, got {type(model).__name__}"
    raise TypeError(err_msg)
model.load(platform, device)

# ── run inference ──────────────────────────────────────────────────────────
prepared = model.prepare(frame)
raw = model.run(prepared)
detections = model.decode(raw, original_size=(frame.shape[0], frame.shape[1]))

# ── draw bounding boxes ────────────────────────────────────────────────────
annotated = frame.copy()
for det in detections:
    x1 = int(det.bbox.x1)
    y1 = int(det.bbox.y1)
    x2 = int(det.bbox.x2)
    y2 = int(det.bbox.y2)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{det.label} {det.confidence:.2f}"
    cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

# ── save result ────────────────────────────────────────────────────────────
output_path = args.out or f"annotated_{Path(args.image).name}"
if not cv2.imwrite(output_path, annotated):
    logger.error("Failed to write output image: %s", output_path)
    raise SystemExit(1)

logger.info("Found %d detection(s). Saved annotated image → %s", len(detections), output_path)

model.unload()
