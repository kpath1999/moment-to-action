"""run_mobileclip_pipeline.py

Runs SensorStage → PreprocessorStage → MobileCLIPStage on an image.

Usage:
    uv run python run_mobileclip_pipeline.py --image weapon.jpg --model mobileclip_s2_datacompdr_last.tflite
"""

import argparse
import logging
import time

from moment_to_action.edgeperceive.core import RawFrameMessage
from moment_to_action.edgeperceive.hardware.types import ComputeUnit
from moment_to_action.edgeperceive.metrics.collector import MetricsCollector
from moment_to_action.edgeperceive.stages import (
    MobileCLIPStage,
    Pipeline,
    PreprocessorStage,
    SensorStage,
)

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
        SensorStage(),
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

msg = RawFrameMessage(frame=None, timestamp=time.time(), source=args.image)
result = pipeline.run(msg)

if result is None:
    print("Pipeline returned no result.")
else:
    print("\nResults:")
    print("-" * 60)
    for prompt, score in sorted(result.all_scores.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 40)
        marker = " ← best" if prompt == result.label else ""
        print(f"  {score:.3f}  {bar:<40}  {prompt}{marker}")
    print("-" * 60)

metrics.print_stage_latencies()
