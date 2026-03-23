"""Runs the YOLO → LLM baseline pipeline on an image.

Moved from ``src/moment_to_action/edgeperceive/pipeline/run_yolo_pipeline.py``.

Usage:
    uv run python scripts/run_yolo_pipeline.py --image weapon.jpg
    uv run python scripts/run_yolo_pipeline.py --image weapon.jpg --device npu
"""

from __future__ import annotations

import argparse
import logging
import time

from rich.console import Console
from rich.logging import RichHandler

from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.messages import RawFrameMessage, ReasoningMessage
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models import ModelManager
from moment_to_action.sensors import FileImageSensor as FileSensor
from moment_to_action.stages import Pipeline
from moment_to_action.stages.llm import LLMStage
from moment_to_action.stages.video import PreprocessorStage, YOLOStage

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
metrics = MetricsCollector()
manager = ModelManager()

# ── build pipeline ─────────────────────────────────────────────────
# Stages resolve their own model paths via ModelManager.
pipeline = Pipeline(
    stages=[
        PreprocessorStage(target_size=(640, 640), letterbox=True),
        YOLOStage(
            backend=ComputeBackend(preferred_unit=device),
            manager=manager,
            confidence_threshold=args.conf,
        ),
        #Replacing the ReasoningStage() with LLMStage()
        #ReasoningStage(),
        #LLMStage(model_path="/home/ubuntu/moment-to-action/llm_models/Qwen3.5-0.8B-Q4_K_M.gguf"),
        LLMStage(model_path="/home/ubuntu/moment-to-action/llm_models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
    ],
    metrics=metrics,
)

# ── load frame via FileSensor, then run pipeline ───────────────────
with FileSensor(args.image) as sensor:
    msg = sensor.read()
    if not isinstance(msg, RawFrameMessage):
        err = f"Expected RawFrameMessage, got {type(msg).__name__}"
        raise TypeError(err)

t_total = time.perf_counter()
result = pipeline.run(msg)
total_ms = (time.perf_counter() - t_total) * 1000

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

metrics.print_stage_latencies()
