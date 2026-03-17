"""Runs the YOLO → LLM baseline pipeline on an image (no metrics).

Usage:
    uv run python run_yolo_pipeline.py --image weapon.jpg --model yolo11n.tflite
    uv run python run_yolo_pipeline.py --image weapon.jpg --model yolo11n.tflite --device npu
"""

import argparse
import logging
import time

from moment_to_action.edgeperceive.core.messages import RawFrameMessage, ReasoningMessage
from moment_to_action.edgeperceive.hardware.types import ComputeUnit
from moment_to_action.edgeperceive.stages import (
    Pipeline,
    PreprocessorStage,
    ReasoningStage,
    SensorStage,
    YOLOStage,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--model", required=True, help="Path to YOLO tflite")
parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
args = parser.parse_args()

device = ComputeUnit.NPU if args.device == "npu" else ComputeUnit.CPU

# ── build pipeline ─────────────────────────────────────────────────
pipeline = Pipeline(
    [
        SensorStage(),
        PreprocessorStage(target_size=(640, 640), letterbox=True),
        YOLOStage(model_path=args.model, confidence_threshold=args.conf, compute_unit=device),
        ReasoningStage(model_path=None),  # LLM stub for now
    ]
)

# ── run ────────────────────────────────────────────────────────────
msg = RawFrameMessage(frame=None, timestamp=time.time(), source=args.image)

t_total = time.perf_counter()
result = pipeline.run(msg)
total_ms = (time.perf_counter() - t_total) * 1000

# ── print results ──────────────────────────────────────────────────
logger.info("\nTotal latency: %.1fms", total_ms)

if result is None:
    logger.info("Pipeline stopped — no detections above threshold.")
elif isinstance(result, ReasoningMessage):
    # Extract detections from the prompt (built by ReasoningStage._build_prompt)
    logger.info("\nYOLO detections:")
    logger.info("-" * 50)
    for line in result.prompt.split("\n"):
        if line.strip().startswith("-"):
            logger.info("%s", line)
    logger.info("-" * 50)
    logger.info("\nLLM response:")
    logger.info("%s", result.response)
    logger.info("\nTotal latency: %.1fms", total_ms)
