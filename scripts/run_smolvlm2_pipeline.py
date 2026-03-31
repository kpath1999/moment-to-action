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
    PYTHONPATH=src uv run python scripts/run_smolvlm2_pipeline.py \
        --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
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
from datetime import UTC, datetime

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
_DEFAULT_OUTPUT_DIR = pathlib.Path("logs/smolvlm2")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_device(value: str) -> int | str:
    """Parse camera index if numeric, otherwise keep as path/URL string."""
    try:
        return int(value)
    except ValueError:
        return value


def _default_run_id() -> str:
    """Return a timestamped run identifier for output artifacts."""
    return datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")


def _append_text_line(path: pathlib.Path, line: str) -> None:
    """Append one line to a text file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        if not line.endswith("\n"):
            f.write("\n")


def _init_markdown_report(
    path: pathlib.Path,
    *,
    run_id: str,
    source: str,
    args: argparse.Namespace,
) -> None:
    """Create a markdown report header for this run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SmolVLM2 Pipeline Run",
        "",
        f"- Run ID: `{run_id}`",
        f"- Source: `{source}`",
        f"- Model: `{args.model}`",
        f"- Device: `{args.torch_device}`",
        f"- clip_len: `{args.clip_len}`",
        f"- stride: `{args.stride}`",
        f"- max_images: `{args.max_images}`",
        f"- max_new_tokens: `{args.max_new_tokens}`",
        "",
        "## Clip Descriptions",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_clip_markdown(
    path: pathlib.Path,
    *,
    clip_count: int,
    frame_count: int,
    latency_ms: float,
    description: str,
) -> None:
    """Append a single full, untruncated clip description to markdown."""
    entry = (
        f"### Clip {clip_count} (frame {frame_count}, stage latency {latency_ms:.1f} ms)\n\n"
        f"{description.strip()}\n\n"
    )
    _append_text_line(path, entry)


def _append_summary_markdown(
    path: pathlib.Path,
    *,
    frame_count: int,
    clip_count: int,
    runtime_s: float,
) -> None:
    """Append a final summary block to markdown report."""
    summary = (
        "## Summary\n\n"
        f"- Total frames processed: `{frame_count}`\n"
        f"- Total clips described: `{clip_count}`\n"
        f"- Runtime: `{runtime_s:.1f}` seconds\n"
    )
    _append_text_line(path, summary)


def _iter_raw_frame_messages(
    video_path: str,
    *,
    max_frames: int = 0,
    progress_every: int = 50,
) -> tuple[int, list[RawFrameMessage]]:
    """Read frames from video file and return RawFrameMessages with progress logs."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        msg = f"Cannot open video file: {video_path!r}"
        raise OSError(msg)

    messages: list[RawFrameMessage] = []
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            frame_count += 1
            messages.append(
                RawFrameMessage(
                    frame=frame,
                    timestamp=time.time(),
                    source=video_path,
                    width=w,
                    height=h,
                )
            )

            if progress_every > 0 and frame_count % progress_every == 0:
                logger.info("Read %d frames from file so far...", frame_count)

            if max_frames > 0 and frame_count >= max_frames:
                break
    finally:
        cap.release()

    return frame_count, messages


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
        default=64,
        help="Maximum generated tokens",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=4,
        help="Maximum sampled frames per clip sent to SmolVLM2",
    )
    parser.add_argument(
        "--clip-len",
        type=int,
        default=32,
        help="Number of frames per buffered clip",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="New frames between clip emissions",
    )
    parser.add_argument(
        "--torch-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="cpu",
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
    parser.add_argument(
        "--progress-every-frames",
        type=int,
        default=16,
        help="Log progress every N processed frames",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory where markdown/log/metrics artifacts are written",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run identifier used in output filenames",
    )
    return parser


# ---------------------------------------------------------------------------
# Frame extraction from file
# ---------------------------------------------------------------------------


def main() -> int:  # noqa: C901, PLR0915
    """Describe a video using SmolVLM2 through the pipeline."""
    args = _build_parser().parse_args()

    run_id = args.run_id.strip() or _default_run_id()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = output_dir / f"{run_id}.log"
    report_path = output_dir / f"{run_id}.md"
    metrics_path = output_dir / f"{run_id}.metrics.json"

    file_handler = logging.FileHandler(raw_log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info("Run ID: %s", run_id)
    logger.info("Detailed logs: %s", raw_log_path)
    logger.info("Markdown report: %s", report_path)
    logger.info("Metrics JSON: %s", metrics_path)

    manager = ModelManager()
    metrics = MetricsCollector(session_id=run_id)

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
    source_label = str(device) if isinstance(device, str) else f"camera:{device}"
    _init_markdown_report(report_path, run_id=run_id, source=source_label, args=args)

    logger.info(
        "Pipeline config: clip_len=%d stride=%d max_images=%d max_new_tokens=%d torch_device=%s",
        args.clip_len,
        args.stride,
        args.max_images,
        args.max_new_tokens,
        args.torch_device,
    )

    frame_count = 0
    clip_count = 0
    last_description = ""
    run_t0 = time.perf_counter()

    if is_file_source:
        logger.info("Extracting frames from video file: %s", device)
        extracted_count, frame_msgs = _iter_raw_frame_messages(
            str(device),
            max_frames=max(0, args.max_frames),
            progress_every=max(1, args.progress_every_frames),
        )
        logger.info("Extracted %d frames, running pipeline ...", extracted_count)
        t0 = time.perf_counter()
        for msg in frame_msgs:
            result = pipeline.run(msg)
            frame_count += 1
            if args.progress_every_frames > 0 and frame_count % args.progress_every_frames == 0:
                logger.info(
                    "Processed %d/%d frames | clips=%d",
                    frame_count,
                    len(frame_msgs),
                    clip_count,
                )
            if isinstance(result, ClassificationMessage):
                clip_count += 1
                last_description = result.label
                logger.info("Clip %d completed at frame %d", clip_count, frame_count)
                logger.info("Clip %d description: %s", clip_count, last_description)
                _append_text_line(
                    raw_log_path,
                    (
                        "FULL_CAPTION "
                        f"clip={clip_count} frame={frame_count} latency_ms={result.latency_ms:.1f} "
                        f"text={last_description}"
                    ),
                )
                _append_clip_markdown(
                    report_path,
                    clip_count=clip_count,
                    frame_count=frame_count,
                    latency_ms=result.latency_ms,
                    description=last_description,
                )
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
                if args.progress_every_frames > 0 and frame_count % args.progress_every_frames == 0:
                    logger.info(
                        "Processed %d frames from stream | clips=%d", frame_count, clip_count
                    )
                if isinstance(result, ClassificationMessage):
                    clip_count += 1
                    last_description = result.label
                    logger.info(
                        "Clip %d @ frame %d",
                        clip_count,
                        frame_count,
                    )
                    logger.info("Clip %d description: %s", clip_count, last_description)
                    _append_text_line(
                        raw_log_path,
                        (
                            "FULL_CAPTION "
                            f"clip={clip_count} frame={frame_count} "
                            f"latency_ms={result.latency_ms:.1f} "
                            f"text={last_description}"
                        ),
                    )
                    _append_clip_markdown(
                        report_path,
                        clip_count=clip_count,
                        frame_count=frame_count,
                        latency_ms=result.latency_ms,
                        description=last_description,
                    )

    if last_description:
        logger.info("\nVideo description:\n%s", last_description)
    else:
        logger.warning("No clips were processed.")

    metrics.print_stage_latencies()
    metrics.save(str(metrics_path))

    runtime_s = time.perf_counter() - run_t0
    _append_summary_markdown(
        report_path,
        frame_count=frame_count,
        clip_count=clip_count,
        runtime_s=runtime_s,
    )
    logger.info("Wrote markdown report to %s", report_path)
    logger.info("Wrote raw log to %s", raw_log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
