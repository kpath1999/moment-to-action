"""Runs the YOLO → LLM baseline pipeline on an image.

Moved from ``src/moment_to_action/edgeperceive/pipeline/run_yolo_pipeline.py``.

Usage:
    uv run python scripts/run_yolo_pipeline.py --image weapon.jpg
    uv run python scripts/run_yolo_pipeline.py --image weapon.jpg --device npu
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import time
import cv2
from pathlib import Path

import rich
from rich.console import Console
from rich.logging import RichHandler

from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.messages import DetectionMessage, ReasoningMessage
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models import ModelManager, ModelID
from moment_to_action.sensors import FileImageSensor as FileSensor
from moment_to_action.stages import Pipeline, ImageSourceStage
from moment_to_action.stages import PromptFormatterStage
from moment_to_action.stages.llm import LLMStage
from moment_to_action.stages.video import PreprocessorStageFrame, YOLOStage

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True, console=Console(stderr=True)),
    ],
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
args = parser.parse_args()

device = ComputeUnit.NPU if args.device == "npu" else ComputeUnit.CPU
#asoma7
#device = ComputeUnit.NPU
compute_backend = ComputeBackend(preferred_unit=device)
metrics = MetricsCollector(
    compute_backend=compute_backend,
    resource_sample_interval=datetime.timedelta(
        seconds=0.01
    ),  # YOLO too fast, need to sample more frequently
)
manager = ModelManager()


def draw_detections(image: np.ndarray, detections: DetectionMessage) -> np.ndarray:
    out = image.copy()
    for box in detections.boxes:
        color = (0, 255, 0) if box.class_id == 0 else (0, 0, 255)
        cv2.rectangle(out, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
        text = f"{box.label} {box.confidence:.2f}"
        cv2.putText(out, text, (int(box.x1), int(box.y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out

# ── build pipeline ─────────────────────────────────────────────────
# Stages resolve their own model paths via ModelManager.
pipeline = Pipeline(
    stages=[
        ImageSourceStage(source_path=args.image),
        PreprocessorStageFrame(target_size=(640, 640), letterbox=True, channels_first=False),
        YOLOStage(
            backend=compute_backend,
            manager=manager,
            confidence_threshold=args.conf,
        ),
        #PromptFormatterStage(
        #    template="json",
        #    min_confidence=0.3,
        #    top_k=5),
        #Replacing the ReasoningStage() with LLMStage()
        #ReasoningStage(),
        #LLMStage(
        #    model_id=ModelID.QWEN_2_5,
        #    manager=manager,
        #),
    ],
)

# ── load frame via FileSensor, then run pipeline ───────────────────
#with FileSensor(args.image) as sensor:
#    msg = sensor.read()

t_total = time.perf_counter()
with metrics.start_trace():
#    result = pipeline.run(msg, metrics=metrics)
    result = pipeline.run(metrics=metrics)
total_ms = (time.perf_counter() - t_total) * 1000

if result and isinstance(result, DetectionMessage):
    frame = cv2.imread(args.image)
    annotated = draw_detections(frame, result)
    cv2.imwrite("result.jpg", annotated)
    print(f"Saved result.jpg with {len(result.boxes)} detection(s)")

# ── print results ──────────────────────────────────────────────────
logger.info("\nTotal latency: %.1fms", total_ms)

if result is None:
    logger.info("Pipeline stopped — no detections above threshold.")
elif isinstance(result, ReasoningMessage):
    logger.info("\nYOLO detections:")
    logger.info("-" * 50)
    for line in result.prompt.split("\n"):
        if line.strip().startswith("-"):
            logger.info("%s", line)
    logger.info("-" * 50)
    logger.info("\nLLM response:")
    logger.info("%s", result.response)

# Log metrics summary
metrics_report = metrics.report()
rich.print("============== Metrics report ==============")
rich.print(metrics_report.summary_full_rich())

with Path("metrics_report.json").open("w") as f:
    json.dump(metrics_report.json(), f, indent=4)
