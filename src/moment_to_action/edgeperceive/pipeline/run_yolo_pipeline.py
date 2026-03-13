"""run_yolo_pipeline.py

Runs the YOLO → LLM baseline pipeline on an image.

Usage:
    uv run python run_yolo_pipeline.py --image weapon.jpg --model yolo11n.tflite
    uv run python run_yolo_pipeline.py --image weapon.jpg --model yolo11n.tflite --device npu
"""

import argparse
import logging
import time

from moment_to_action.edgeperceive.core import RawFrameMessage
from moment_to_action.edgeperceive.hardware.types import ComputeUnit
from moment_to_action.edgeperceive.metrics.collector import MetricsCollector
from moment_to_action.edgeperceive.stages import (
    # Pipeline, RawFrameMessage,
    Pipeline,
    PreprocessorStage,
    ReasoningStage,
    SensorStage,
    YOLOStage,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--model", required=True, help="Path to YOLO tflite")
parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
args = parser.parse_args()

device = ComputeUnit.NPU if args.device == "npu" else ComputeUnit.CPU
metrics = MetricsCollector()

# ── build pipeline ─────────────────────────────────────────────────
# build a pipeline as a sequence of stages
pipeline = Pipeline(
    stages=[
        SensorStage(),
        PreprocessorStage(target_size=(640, 640), letterbox=True),
        YOLOStage(model_path=args.model, confidence_threshold=args.conf, compute_unit=device),
        ReasoningStage(model_path=None),
    ],
    metrics=metrics,
)

# ── run ────────────────────────────────────────────────────────────
msg = RawFrameMessage(frame=None, timestamp=time.time(), source=args.image)

t_total = time.perf_counter()
result = pipeline.run(msg)
total_ms = (time.perf_counter() - t_total) * 1000

# ── print results ──────────────────────────────────────────────────
print(f"\nTotal latency: {total_ms:.1f}ms")

if result is None:
    print("Pipeline stopped — no detections above threshold.")
else:
    print("\nYOLO detections:")
    print("-" * 50)
    for line in result.prompt.split("\n"):
        if line.strip().startswith("-"):
            print(line)
    print("-" * 50)
    print("\nLLM response:")
    print(result.response)

metrics.print_stage_latencies()
