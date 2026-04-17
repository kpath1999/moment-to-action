"""Runs YAMNet → LLM baseline pipeline on an audio recording.

Usage:
    uv run python scripts/run_yamnet_pipeline.py --audio audio_recording.wav
    uv run python scripts/run_yamnet_pipeline.py --audio audio_recording.wav --device npu
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import time
from pathlib import Path

import numpy as np
import rich
from rich.console import Console
from rich.logging import RichHandler

from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.messages import ReasoningMessage
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models import ModelManager, ModelID
from moment_to_action.sensors import FileImageSensor as FileSensor
from moment_to_action.messages import AudioInput
from moment_to_action.stages import Pipeline
from moment_to_action.stages import PromptFormatterStage
from moment_to_action.stages.llm import LLMStage
from moment_to_action.stages.audio import YAMNetPreprocessorStage, YAMNetStage

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
parser.add_argument("--audio", required=True)
parser.add_argument("--device", choices=["cpu", "npu"], default="cpu")
parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
args = parser.parse_args()

device = ComputeUnit.NPU if args.device == "npu" else ComputeUnit.CPU
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
        YAMNetPreprocessorStage(),
        YAMNetStage(
            backend=compute_backend,
            manager=manager,
        #    class_names=YAMNET_LABELS,
            confidence_threshold=0.3,
            aggregation="mean",
        ),
        #PreprocessorStage(target_size=(640, 640), letterbox=True),
        #PromptFormatterStage(
        #    template="json",
        #    min_confidence=0.3,
        #    top_k=5),
        #Replacing the ReasoningStage() with LLMStage()
        #ReasoningStage(),
        #LLMStage(model_path="/home/ubuntu/moment-to-action/llm_models/Qwen3.5-0.8B-Q4_K_M.gguf"),
        #LLMStage(model_path="/home/ubuntu/moment-to-action/llm_models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
        #LLMStage(
        #    model_id=ModelID.QWEN_2_5,
        #    manager=manager,
        #),
    ],
)

#waveform = np.zeros(16000, dtype=np.float32)
sample_rate = 16000
duration_s = 1.0
frequency_hz = 2000.0

t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
waveform = 0.5 * np.sin(2 * np.pi * frequency_hz * t).astype(np.float32)

msg = AudioInput(
        waveform=waveform,
        source="mic0",
        sample_rate=16000,
        num_samples=len(waveform),
        timestamp=time.time(),
    )

t_start = time.perf_counter();
with metrics.start_trace():
    result = pipeline.run(msg, metrics=metrics)
total_ms = time.perf_counter() - t_start;

print(result)



# ── print results ──────────────────────────────────────────────────
logger.info("\nTotal latency: %.1fms", total_ms)

"""
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
"""
# Log metrics summary
metrics_report = metrics.report()
rich.print("============== Metrics report ==============")
rich.print(metrics_report.summary_full_rich())

with Path("metrics_report.json").open("w") as f:
    json.dump(metrics_report.json(), f, indent=4)
