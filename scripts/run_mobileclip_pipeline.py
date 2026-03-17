"""Runs PreprocessorStage → MobileCLIPStage on an image.

Moved from ``src/moment_to_action/edgeperceive/pipeline/run_mobileclip_pipeline.py``.
Frame loading is now handled by ``FileSensor`` before entering the pipeline.

Usage::

    uv run python scripts/run_mobileclip_pipeline.py --image weapon.jpg \
        --model mobileclip_s2_datacompdr_last.tflite
"""

from __future__ import annotations

import argparse
import logging

from moment_to_action.hardware import ComputeUnit
from moment_to_action.messages import ClassificationMessage, RawFrameMessage
from moment_to_action.metrics.collector import MetricsCollector
from moment_to_action.sensors import FileSensor
from moment_to_action.stages import (
    MobileCLIPStage,
    Pipeline,
    PreprocessorStage,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--model", required=True)
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
metrics = MetricsCollector()

pipeline = Pipeline(
    stages=[
        PreprocessorStage(
            target_size=(256, 256),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            letterbox=False,
        ),
        MobileCLIPStage(
            model_path=args.model,
            text_prompts=PROMPTS,
            compute_unit=device,
        ),
    ],
    metrics=metrics,
)

# ── load frame via FileSensor, then run pipeline ───────────────────
with FileSensor(args.image) as sensor:
    msg: RawFrameMessage = sensor.read()

result = pipeline.run(msg)

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

metrics.print_stage_latencies()
