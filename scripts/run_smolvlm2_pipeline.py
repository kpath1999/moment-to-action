r"""Describe a video using SmolVLM2 via the moment-to-action pipeline.

Runs SmolVLM2 on a video file or camera stream using the standard pipeline:

    CameraStreamSensor / file frames
        → ClipBufferStage (accumulate frames into clips)
        → SmolVLM2Stage   (VLM inference → ClassificationMessage)

If the model is not present locally, this script downloads it from Hugging Face
into the configured model cache directory. Authentication supports ``HF_TOKEN``.

Examples:
    # Describe a video file
    PYTHONPATH=src uv run python scripts/run_smolvlm2_pipeline.py \\
        --device images/smoke_test.mp4 \\
        --max-new-tokens 256

    # Use explicit model override (default is SmolVLM2-2.2B-Instruct)
    PYTHONPATH=src uv run python scripts/run_smolvlm2_pipeline.py \\
        --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \\
        --device images/smoke_test.mp4

    # Webcam capture and describe
    PYTHONPATH=src uv run python scripts/run_smolvlm2_pipeline.py \\
        --device 0 --max-frames 160
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import signal
import time

import cv2

from moment_to_action.messages import ClassificationMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models import ModelManager
from moment_to_action.sensors import CameraStreamSensor
from moment_to_action.stages import Pipeline
from moment_to_action.stages.video import ClipBufferStage
from moment_to_action.stages.vlm import SmolVLM2Stage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

_DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
_DEFAULT_PROMPT = "Describe this video in detail"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_device(value: str) -> int | str:
    """Parse camera index if numeric, otherwise keep as path/URL string."""
    try:
        return int(value)
    except ValueError:
        return value


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Describe a video using SmolVLM2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL_ID,
        help="HF model id or local model directory",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Camera index (e.g. 0) or video file/URL path",
    )
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT, help="User prompt for SmolVLM2")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=8,
        help="Maximum sampled frames per clip sent to SmolVLM2",
    )
    parser.add_argument(
        "--clip-len",
        type=int,
        default=16,
        help="Number of frames per buffered clip",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="New frames between clip emissions",
    )
    parser.add_argument(
        "--torch-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Torch execution device",
    )
    parser.add_argument("--width", type=int, default=0, help="Requested capture width (camera)")
    parser.add_argument("--height", type=int, default=0, help="Requested capture height (camera)")
    parser.add_argument("--fps", type=float, default=0.0, help="Requested capture fps (camera)")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames; 0 means unlimited",
    )
    return parser


# ---------------------------------------------------------------------------
# Frame extraction from file
# ---------------------------------------------------------------------------


def _extract_raw_frame_messages(video_path: str, max_frames: int = 0) -> list[RawFrameMessage]:
    """Read all (or up to max_frames) frames from a video file as RawFrameMessages."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        msg = f"Cannot open video file: {video_path!r}"
        raise OSError(msg)
    messages: list[RawFrameMessage] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            messages.append(
                RawFrameMessage(
                    frame=frame,
                    timestamp=time.time(),
                    source=video_path,
                    width=w,
                    height=h,
                )
            )
            if max_frames > 0 and len(messages) >= max_frames:
                break
    finally:
        cap.release()
    return messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Describe a video using SmolVLM2 through the pipeline."""
    args = _build_parser().parse_args()

    manager = ModelManager()
    metrics = MetricsCollector()

    # Build the pipeline: ClipBufferStage → SmolVLM2Stage
    pipeline = Pipeline(
        stages=[
            ClipBufferStage(clip_len=args.clip_len, stride=args.stride),
            SmolVLM2Stage(
                model_name=args.model,
                cache_dir=str(manager.cache_dir),
                torch_device=args.torch_device,
                prompt=args.prompt,
                max_images=args.max_images,
                max_new_tokens=args.max_new_tokens,
            ),
        ],
        metrics=metrics,
    )

    device = _parse_device(args.device)
    is_file_source = isinstance(device, str) and pathlib.Path(device).exists()

    frame_count = 0
    clip_count = 0
    last_description = ""

    if is_file_source:
        logger.info("Extracting frames from video file: %s", device)
        frame_msgs = _extract_raw_frame_messages(str(device), max_frames=max(0, args.max_frames))
        logger.info("Extracted %d frames, running pipeline ...", len(frame_msgs))
        t0 = time.perf_counter()
        for msg in frame_msgs:
            result = pipeline.run(msg)
            frame_count += 1
            if isinstance(result, ClassificationMessage):
                clip_count += 1
                last_description = result.label
                logger.info("Clip %d: %s", clip_count, last_description[:120])
        latency_s = time.perf_counter() - t0
        logger.info("Pipeline completed in %.1fs — %d clips processed", latency_s, clip_count)
    else:
        # Camera input: stream frames through the pipeline.
        running = True

        def _handle_sigint(_sig: int, _frame: object) -> None:
            nonlocal running
            logger.info("Interrupted — shutting down...")
            running = False

        signal.signal(signal.SIGINT, _handle_sigint)

        logger.info(
            "Streaming from device=%r (clip_len=%d, stride=%d)",
            device,
            args.clip_len,
            args.stride,
        )
        with CameraStreamSensor(
            device=device,
            width=args.width,
            height=args.height,
            fps=args.fps,
        ) as sensor:
            while running:
                if args.max_frames > 0 and frame_count >= args.max_frames:
                    logger.info("Reached --max-frames %d", args.max_frames)
                    break
                msg = sensor.read()
                if msg.frame is None:
                    continue
                result = pipeline.run(msg)
                frame_count += 1
                if isinstance(result, ClassificationMessage):
                    clip_count += 1
                    last_description = result.label
                    logger.info(
                        "Clip %d @ frame %d: %s",
                        clip_count,
                        frame_count,
                        last_description[:120],
                    )

    if last_description:
        logger.info("\nVideo description:\n%s", last_description)
    else:
        logger.warning("No clips were processed.")

    metrics.print_stage_latencies()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
