"""Live video SmolVLM2 pipeline.

Runs an end-to-end flow:

    CameraStreamSensor
        -> ClipBufferStage
        -> SmolVLM2 inference (Transformers)
        -> text output per clip

If the model is not present locally, this script downloads it from Hugging Face
into the configured model cache directory. Authentication supports `HF_TOKEN`.

Examples:
    # Webcam
    PYTHONPATH=src uv run python scripts/run_camera_smolvlm2_pipeline.py \
        --device 0 \
        --show

    # Video file (smoke test)
    PYTHONPATH=src uv run python scripts/run_camera_smolvlm2_pipeline.py \
        --device images/smoke_test.mp4 \
        --clip-len 16 --stride 16
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import signal
import time

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from moment_to_action.messages import RawFrameMessage, VideoClipMessage
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models import ModelManager
from moment_to_action.sensors import CameraStreamSensor
from moment_to_action.stages import Pipeline
from moment_to_action.stages.video import ClipBufferStage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


_DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
_DEFAULT_PROMPT = "Describe the key action happening in these frames."
_DEFAULT_SYSTEM = (
    "Focus on salient action and safety-relevant events. "
    "Be concise and factual."
)


def _draw_caption(frame: object, caption: str) -> object:
    """Overlay the latest SmolVLM2 caption on a frame."""
    out = frame.copy()
    text = caption.strip() or "(no caption yet)"
    lines = [text[i : i + 80] for i in range(0, len(text), 80)][:3]
    y = 28
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        y += 24

    return out


def _parse_device(value: str) -> int | str:
    """Parse camera index if numeric, otherwise keep as path/URL string."""
    try:
        return int(value)
    except ValueError:
        return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SmolVLM2 over webcam/video clips and print captions.",
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
    parser.add_argument("--system", default=_DEFAULT_SYSTEM, help="System prompt for SmolVLM2")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Maximum generated tokens for each clip",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=8,
        help="Maximum sampled frames per clip sent to SmolVLM2",
    )
    parser.add_argument("--clip-len", type=int, default=16, help="Frames per buffered clip")
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
    parser.add_argument("--width", type=int, default=0, help="Requested capture width")
    parser.add_argument("--height", type=int, default=0, help="Requested capture height")
    parser.add_argument("--fps", type=float, default=0.0, help="Requested capture fps")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live OpenCV window with rendered detections",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 means unlimited)",
    )
    return parser


def _select_torch_device(requested: str) -> torch.device:
    """Select the best available torch device for inference."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _select_torch_dtype(device: torch.device) -> torch.dtype:
    """Choose a safe dtype for the selected device."""
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _sample_frames(frames: list[object], max_images: int) -> list[object]:
    """Uniformly sample up to max_images frames from the clip."""
    if len(frames) <= max_images:
        return frames
    if max_images < 1:
        msg = "max_images must be >= 1"
        raise ValueError(msg)
    step = (len(frames) - 1) / (max_images - 1)
    indices = [round(i * step) for i in range(max_images)]
    return [frames[idx] for idx in indices]


def _build_messages(
    system_prompt: str,
    user_prompt: str,
    images: list[Image.Image],
) -> list[dict[str, object]]:
    """Build a multi-image chat payload for SmolVLM2."""
    user_content: list[dict[str, object]] = [{"type": "text", "text": user_prompt}]
    user_content.extend({"type": "image", "image": image} for image in images)
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


def _clean_generation(decoded: str) -> str:
    """Strip wrapper chat text and keep only the assistant answer."""
    marker = "Assistant:"
    if marker in decoded:
        return decoded.rsplit(marker, maxsplit=1)[-1].strip()
    return decoded.strip()


def _to_pil_rgb(bgr_frame: object) -> Image.Image:
    """Convert an OpenCV BGR frame into a PIL RGB image."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _load_smolvlm2(
    model_name: str,
    cache_dir: pathlib.Path,
    torch_device: torch.device,
    torch_dtype: torch.dtype,
) -> tuple[AutoProcessor, AutoModelForImageTextToText]:
    """Load SmolVLM2 processor/model, downloading from HF cache when needed."""
    logger.info(
        "Loading SmolVLM2 model=%s (device=%s, dtype=%s)",
        model_name,
        torch_device,
        torch_dtype,
    )
    logger.info("Model cache dir: %s", cache_dir)

    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(torch_device)
    model.eval()
    return processor, model


def _infer_clip(
    clip_msg: VideoClipMessage,
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    system_prompt: str,
    user_prompt: str,
    max_images: int,
    max_new_tokens: int,
) -> tuple[str, float]:
    """Run SmolVLM2 on sampled clip frames and return caption plus latency."""
    sampled_frames = _sample_frames(clip_msg.frames, max_images)
    images = [_to_pil_rgb(frame) for frame in sampled_frames]
    messages = _build_messages(system_prompt, user_prompt, images)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

    t_start = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    inference_ms = (time.perf_counter() - t_start) * 1000.0

    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    caption = _clean_generation(decoded[0]) if decoded else ""
    return caption, inference_ms


def main() -> int:
    """Run live-video SmolVLM2 inference and print clip-level captions."""
    args = _build_parser().parse_args()

    manager = ModelManager()
    torch_device = _select_torch_device(args.torch_device)
    torch_dtype = _select_torch_dtype(torch_device)
    processor, model = _load_smolvlm2(
        model_name=args.model,
        cache_dir=manager.cache_dir,
        torch_device=torch_device,
        torch_dtype=torch_dtype,
    )

    metrics = MetricsCollector()
    pipeline = Pipeline(
        stages=[
            ClipBufferStage(clip_len=args.clip_len, stride=args.stride),
        ],
        metrics=metrics,
    )

    running = True

    def _handle_sigint(_sig: int, _frame: object) -> None:
        nonlocal running
        logger.info("Interrupted, shutting down...")
        running = False

    signal.signal(signal.SIGINT, _handle_sigint)

    device = _parse_device(args.device)
    frame_count = 0
    clip_count = 0
    last_caption = ""

    logger.info(
        "Starting camera SmolVLM2 pipeline: device=%r clip_len=%d stride=%d",
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

            msg: RawFrameMessage = sensor.read()
            frame_count += 1

            result = pipeline.run(msg)
            preview = msg.frame

            if isinstance(result, VideoClipMessage):
                clip_count += 1
                last_caption, inference_ms = _infer_clip(
                    clip_msg=result,
                    processor=processor,
                    model=model,
                    system_prompt=args.system,
                    user_prompt=args.prompt,
                    max_images=args.max_images,
                    max_new_tokens=args.max_new_tokens,
                )
                logger.info(
                    "Clip %d @ frame %d [%.1fms]: %s",
                    clip_count,
                    frame_count,
                    inference_ms,
                    last_caption,
                )

            if args.show and preview is not None:
                preview = _draw_caption(preview, last_caption)
                cv2.imshow("moment-to-action | smolvlm2", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Preview window closed, shutting down.")
                    break

    if args.show:
        cv2.destroyAllWindows()

    logger.info("Run summary: frames=%d, clips_processed=%d", frame_count, clip_count)
    metrics.print_stage_latencies()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
