r"""Live-camera temporal action pipeline.

Demonstrates the full video pipeline with live camera input:

    CameraStreamSensor
        down RawFrameMessage (per-frame)
    ClipBufferStage        <- accumulates frames -> VideoClipMessage
        down VideoClipMessage (every `stride` frames once buffer fills)
    TemporalActionStage    <- temporal action recognition
        down ActionMessage

Usage:
    # Webcam (device 0), MoViNet model, sliding window of 16 frames::

        uv run python scripts/run_camera_action_pipeline.py \\
            --model models/movinet_a0.tflite \\
            --clip-len 16 --stride 8

    # Video file instead of live camera::

        uv run python scripts/run_camera_action_pipeline.py \\
            --device path/to/video.mp4 \\
            --model models/movinet_a0.tflite \\
            --clip-len 16 --stride 16

    # Use NPU, top-5 predictions, custom label file::

        uv run python scripts/run_camera_action_pipeline.py \\
            --model models/action_model.tflite \\
            --labels kinetics400_labels.txt \\
            --device-hw npu --top-k 5

Notes:
    - Press ``q`` in the OpenCV preview window (if ``--show``) or Ctrl-C to stop.
    - When ``--show`` is set, detections are overlaid on the live feed.
    - The script is intentionally dependency-light; it only requires the
      packages already listed in ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import signal
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── argument parsing ───────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Live / video-file temporal action recognition pipeline.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--device",
    default="0",
    help="Camera device index (int) or video-file path / stream URL.",
)
parser.add_argument(
    "--model",
    required=True,
    help="Path to the temporal action model (TFLite or ONNX).",
)
parser.add_argument(
    "--clip-profile",
    choices=["movinet", "videomae"],
    default="movinet",
    help="Clip pre-processing profile matching the model architecture.",
)
parser.add_argument(
    "--clip-len",
    type=int,
    default=16,
    help="Number of frames per clip fed to the temporal model.",
)
parser.add_argument(
    "--stride",
    type=int,
    default=None,
    help="New frames between each emission. Default = clip-len (non-overlapping).",
)
parser.add_argument(
    "--labels",
    default=None,
    help="Path to a text file with one action label per line (class-ID order).",
)
parser.add_argument(
    "--top-k",
    type=int,
    default=3,
    help="Number of top action predictions to log.",
)
parser.add_argument(
    "--conf",
    type=float,
    default=0.0,
    help="Minimum confidence threshold for predictions.",
)
parser.add_argument(
    "--device-hw",
    choices=["cpu", "npu", "gpu"],
    default="cpu",
    help="Compute unit for inference.",
)
parser.add_argument(
    "--width",
    type=int,
    default=0,
    help="Requested capture width (0 = device default).",
)
parser.add_argument(
    "--height",
    type=int,
    default=0,
    help="Requested capture height (0 = device default).",
)
parser.add_argument(
    "--fps",
    type=float,
    default=0.0,
    help="Requested capture frame-rate (0 = device default).",
)
parser.add_argument(
    "--min-fps",
    type=float,
    default=0.0,
    help="Discard clips with effective fps below this value (0 = disabled).",
)
parser.add_argument(
    "--show",
    action="store_true",
    help="Display a live OpenCV preview window with the top action overlaid.",
)
parser.add_argument(
    "--max-frames",
    type=int,
    default=0,
    help="Stop after capturing this many frames (0 = unlimited).",
)
args = parser.parse_args()

# ── imports (after arg parse so --help is fast) ────────────────────
from moment_to_action.hardware import ComputeUnit  # noqa: E402
from moment_to_action.messages import ActionMessage, RawFrameMessage  # noqa: E402
from moment_to_action.metrics import MetricsCollector  # noqa: E402
from moment_to_action.sensors import CameraStreamSensor  # noqa: E402
from moment_to_action.stages import Pipeline  # noqa: E402
from moment_to_action.stages.video import ClipBufferStage, TemporalActionStage  # noqa: E402

# ── load optional labels ───────────────────────────────────────────
labels: list[str] | None = None
if args.labels:
    label_path = pathlib.Path(args.labels)
    if not label_path.is_file():
        logger.error("Label file not found: %s", label_path)
        sys.exit(1)
    labels = label_path.read_text().splitlines()
    logger.info("Loaded %d action labels from %s", len(labels), label_path)

# ── map compute unit ───────────────────────────────────────────────
_HW_MAP: dict[str, ComputeUnit] = {
    "cpu": ComputeUnit.CPU,
    "npu": ComputeUnit.NPU,
    "gpu": ComputeUnit.GPU,
}
compute_unit = _HW_MAP[args.device_hw]

# ── device index ───────────────────────────────────────────────────
# Try to parse as int; fall back to string (file path / URL).
try:
    capture_device: int | str = int(args.device)
except ValueError:
    capture_device = args.device

# ── metrics ────────────────────────────────────────────────────────
metrics = MetricsCollector()

# ── build pipeline ─────────────────────────────────────────────────
pipeline = Pipeline(
    stages=[
        ClipBufferStage(
            clip_len=args.clip_len,
            stride=args.stride,  # None → default (= clip_len)
            min_fps=args.min_fps,
        ),
        TemporalActionStage(
            model_path=args.model,
            clip_profile=args.clip_profile,
            labels=labels,
            top_k=args.top_k,
            confidence_threshold=args.conf,
            compute_unit=compute_unit,
        ),
    ],
    metrics=metrics,
)

# ── graceful shutdown ──────────────────────────────────────────────
_running = True


def _handle_sigint(sig: int, frame: object) -> None:  # noqa: ARG001
    global _running  # noqa: PLW0603
    logger.info("Interrupted — shutting down.")
    _running = False


signal.signal(signal.SIGINT, _handle_sigint)

# ── main loop ──────────────────────────────────────────────────────
logger.info(
    "Starting pipeline: device=%r  clip_len=%d  stride=%s  profile=%s  hw=%s",
    capture_device,
    args.clip_len,
    args.stride or args.clip_len,
    args.clip_profile,
    args.device_hw,
)

frame_count = 0
last_action: str = "—"

with CameraStreamSensor(
    device=capture_device,
    width=args.width,
    height=args.height,
    fps=args.fps,
) as sensor:
    while _running:
        msg: RawFrameMessage = sensor.read()

        if args.max_frames > 0 and frame_count >= args.max_frames:
            logger.info("Reached --max-frames %d, stopping.", args.max_frames)
            break

        result = pipeline.run(msg)
        frame_count += 1

        if isinstance(result, ActionMessage) and result.top_action is not None:
            top = result.top_action
            last_action = f"{top.label} ({top.confidence:.2f})"
            logger.info(
                "Action [frame %d]: %s  (latency=%.1fms)",
                frame_count,
                last_action,
                result.latency_ms,
            )
            if args.top_k > 1:
                for pred in result.predictions[1:]:
                    logger.info("  ↳ %s (%.2f)", pred.label, pred.confidence)

        # Optional preview window — overlay the current top action.
        if args.show and msg.frame is not None:
            import cv2

            preview = msg.frame.copy()
            cv2.putText(
                preview,
                last_action,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("moment-to-action | live", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Preview window closed — shutting down.")
                break

    if args.show:
        import cv2

        cv2.destroyAllWindows()

# ── summary ────────────────────────────────────────────────────────
logger.info("\nTotal frames captured: %d", frame_count)
metrics.print_stage_latencies()
