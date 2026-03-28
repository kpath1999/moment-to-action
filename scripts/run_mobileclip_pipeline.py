"""Runs PreprocessorStage → MobileCLIPStage on an image.

Moved from ``src/moment_to_action/edgeperceive/pipeline/run_mobileclip_pipeline.py``.
Frame loading is now handled by ``FileSensor`` before entering the pipeline.

Usage::

    uv run python scripts/run_mobileclip_pipeline.py --image weapon.jpg
"""

from __future__ import annotations

import argparse
import logging

from rich.console import Console
from rich.logging import RichHandler

from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.messages import ClassificationMessage, RawFrameMessage, ReasoningMessage
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models import ModelManager
from moment_to_action.sensors import FileImageSensor as FileSensor
from moment_to_action.stages import Pipeline
from moment_to_action.stages.video import PreprocessorStage
from moment_to_action.stages.vlm import MobileCLIPStage
from moment_to_action.stages.llm import LLMStage

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
metrics = MetricsCollector()
manager = ModelManager()

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
            backend=ComputeBackend(preferred_unit=device),
            manager=manager,
        ),
        #LLMStage(model_path="/home/ubuntu/moment-to-action/llm_models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
        LLMStage(model_path="/home/ubuntu/moment-to-action/llm_models/qwen2.5-1.5b-instruct-q5_k_m.gguf"),
    ],
    metrics=metrics,
)

# ── load frame via FileSensor, then run pipeline ───────────────────
with FileSensor(args.image) as sensor:
    msg = sensor.read()
    if not isinstance(msg, RawFrameMessage):
        err = f"Expected RawFrameMessage, got {type(msg).__name__}"
        raise TypeError(err)

result = pipeline.run(msg)

if result is None:
    logger.info("Pipeline returned no result.")
elif isinstance(result, ReasoningMessage):
    logger.info("\nResults:")
    logger.info("-" * 60)
    '''
    for prompt, score in sorted(result.all_scores.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 40)
        marker = " ← best" if prompt == result.label else ""
        logger.info("  %.3f  %-40s  %s%s", score, bar, prompt, marker)
    '''
    logger.info("-" * 60)
    logger.info("\nLLM response:")
    logger.info("%s", result.response)

metrics.print_stage_latencies()
