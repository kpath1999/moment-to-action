"""Runs PreprocessorStage → MobileCLIPStage on an image.

Moved from ``src/moment_to_action/edgeperceive/pipeline/run_mobileclip_pipeline.py``.
Frame loading is now handled by ``FileSensor`` before entering the pipeline.

Usage::

    uv run python scripts/run_mobileclip_pipeline.py --image weapon.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import rich
from rich.console import Console
from rich.logging import RichHandler

from moment_to_action.hardware import ComputeUnit, Platform
from moment_to_action.messages import ClassificationMessage
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models import ModelManager
from moment_to_action.paths import PathManager
from moment_to_action.sensors import FileImageSensor as FileSensor
from moment_to_action.stages import Pipeline
from moment_to_action.stages.video import PreprocessorStage
from moment_to_action.stages.vlm import MobileCLIPStage

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
args = parser.parse_args()

PROMPTS = [
    "a photo of a person holding a gun",
    "a photo of a person walking normally",
    "a photo of people having fun",
    "a photo of a fight or physical altercation",
    "a photo of a person in distress",
]

device = ComputeUnit.NPU if args.device == "npu" else ComputeUnit.CPU
platform = Platform()
metrics = MetricsCollector(compute_platform=platform)
manager = ModelManager(PathManager())

pipeline = Pipeline(
    stages=[
        PreprocessorStage(
            target_size=(256, 256),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            letterbox=False,
        ),
        # Stage resolves the MobileCLIP model path via ModelManager.
        MobileCLIPStage(
            text_prompts=PROMPTS,
            platform=platform,
            unit=device,
            manager=manager,
        ),
    ],
)

# ── load frame via FileSensor, then run pipeline ───────────────────
with FileSensor(args.image) as sensor:
    msg = sensor.read()

with metrics.start_trace():
    result = pipeline.run(msg, metrics=metrics)

if result is None:
    logger.info("Pipeline returned no result.")
elif isinstance(result, ClassificationMessage):
    logger.info("\nResults:")
    logger.info("-" * 60)
    for prompt, score in sorted(result.all_scores.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 40)
        marker = " ← best" if prompt == result.label else ""
        logger.info("  %.3f  %-40s  %s%s", score, bar, prompt, marker)
    logger.info("-" * 60)

# Log metrics summary
metrics_report = metrics.report()
rich.print("============== Metrics report ==============")
rich.print(metrics_report.summary_full_rich())

with Path("metrics_report.json").open("w") as f:
    json.dump(metrics_report.json(), f, indent=4)
