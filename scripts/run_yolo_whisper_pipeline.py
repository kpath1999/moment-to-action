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
from pathlib import Path

import rich
from rich.console import Console
from rich.logging import RichHandler

from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.messages import ReasoningMessage
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models import ModelManager, ModelID
from moment_to_action.sensors import FileImageSensor as FileSensor
from moment_to_action.stages import Pipeline, ImageSourceStage, AudioSourceStage
from moment_to_action.stages import PromptFormatterStage
from moment_to_action.stages.llm import LLMStage
from moment_to_action.stages.video import PreprocessorStage, YOLOStage
from moment_to_action.stages.audio import WhisperPreprocessorStage, WhisperStage
from moment_to_action.stages import TriggerStage

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
parser.add_argument("--audio", required=True)
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


# ── build pipeline ─────────────────────────────────────────────────
# Stages resolve their own model paths via ModelManager.
pipeline = Pipeline(
    stages=[
        ImageSourceStage(source_path=args.image),
        PreprocessorStage(target_size=(640, 640), letterbox=True),
        YOLOStage(
            #backend=compute_backend,
            backend=ComputeBackend(preferred_unit=ComputeUnit.NPU),
            manager=manager,
            confidence_threshold=args.conf,
        ),
        TriggerStage(),
        AudioSourceStage(source_path=args.audio),
        WhisperPreprocessorStage(),
        WhisperStage(
            model_size_or_path="small",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            vad_filter=False,
        ),        
        TriggerStage(),
        PromptFormatterStage(
            template="json",
            min_confidence=0.3,
            top_k=5),
        #Replacing the ReasoningStage() with LLMStage()
        LLMStage(
            model_id=ModelID.QWEN_2_5,
            manager=manager,
        ),
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
