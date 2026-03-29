"""SmolVLM2 video-description stage.

SmolVLM2Stage runs SmolVLM2 (a vision-language model) over sampled frames
from a :class:`~moment_to_action.messages.video.VideoClipMessage` and
produces a :class:`~moment_to_action.messages.vlm.ClassificationMessage`
with a natural-language scene description.

This stage sits at the same level as YOLO / MobileCLIP in the pipeline —
its output feeds into the downstream reasoning stage.

Input:  VideoClipMessage
Output: ClassificationMessage
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from moment_to_action.messages.video import VideoClipMessage
from moment_to_action.messages.vlm import ClassificationMessage
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    import numpy as np

    from moment_to_action.messages import Message

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
_DEFAULT_PROMPT = "Describe the key action happening in these frames."
_DEFAULT_SYSTEM = "Focus on salient action and safety-relevant events. Be concise and factual."


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


def _to_pil_rgb(bgr_frame: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR frame into a PIL RGB image."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _sample_frames(frames: list[np.ndarray], max_images: int) -> list[np.ndarray]:
    """Uniformly sample up to *max_images* frames, preserving temporal order."""
    if len(frames) <= max_images:
        return frames
    step = (len(frames) - 1) / (max_images - 1)
    indices = [round(i * step) for i in range(max_images)]
    return [frames[idx] for idx in indices]


class SmolVLM2Stage(Stage):
    """Run SmolVLM2 on sampled video-clip frames and produce a scene description.

    The model is loaded once at construction via HuggingFace Transformers
    and reused across all calls.  Frames are uniformly sampled from the
    incoming :class:`VideoClipMessage`, converted to PIL images, and fed
    through the SmolVLM2 chat template.

    Args:
        model_name: HuggingFace model identifier or local directory.
        cache_dir: Directory for HuggingFace model cache.  When ``None``,
            uses the default ``~/.cache/huggingface`` location.
        torch_device: ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
        prompt: User prompt describing what to look for.
        system_prompt: System-level instruction for the model.
        max_images: Maximum frames sampled per clip.
        max_new_tokens: Generation length limit.

    Input:  :class:`~moment_to_action.messages.video.VideoClipMessage`
    Output: :class:`~moment_to_action.messages.vlm.ClassificationMessage`
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_ID,
        cache_dir: str | None = None,
        torch_device: str = "auto",
        prompt: str = _DEFAULT_PROMPT,
        system_prompt: str = _DEFAULT_SYSTEM,
        max_images: int = 8,
        max_new_tokens: int = 96,
    ) -> None:
        super().__init__()
        self._prompt = prompt
        self._system_prompt = system_prompt
        self._max_images = max(1, max_images)
        self._max_new_tokens = max_new_tokens

        device = _select_torch_device(torch_device)
        dtype = _select_torch_dtype(device)

        cache_kwargs: dict[str, str] = {}
        if cache_dir is not None:
            cache_kwargs["cache_dir"] = cache_dir

        logger.info("SmolVLM2Stage: loading %s (device=%s, dtype=%s)", model_name, device, dtype)
        self._processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            **cache_kwargs,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
            **cache_kwargs,
        ).to(device)  # type: ignore[arg-type]
        self._model.eval()
        logger.info("SmolVLM2Stage: ready")

    # ------------------------------------------------------------------
    # Stage interface
    # ------------------------------------------------------------------

    def _process(self, msg: Message) -> ClassificationMessage | None:
        """Sample frames from a clip, run SmolVLM2, and return a description."""
        if not isinstance(msg, VideoClipMessage):
            type_name = type(msg).__name__
            err = f"SmolVLM2Stage expects VideoClipMessage, got {type_name}"
            raise TypeError(err)

        sampled = _sample_frames(msg.frames, self._max_images)  # type: ignore[arg-type]
        logger.info(
            "SmolVLM2Stage: sampled %d/%d frames (max_images=%d)",
            len(sampled),
            len(msg.frames),
            self._max_images,
        )

        t_prepare = time.perf_counter()
        images = [_to_pil_rgb(f) for f in sampled]

        user_content: list[dict[str, object]] = [{"type": "text", "text": self._prompt}]
        user_content.extend({"type": "image", "image": img} for img in images)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": [{"type": "text", "text": self._system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        prepare_ms = (time.perf_counter() - t_prepare) * 1000.0
        logger.info(
            "SmolVLM2Stage: prepare %.1fms (device=%s, prompt_tokens=%d, max_new_tokens=%d)",
            prepare_ms,
            self._model.device,
            input_len,
            self._max_new_tokens,
        )

        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self._max_new_tokens,
                )
        except RuntimeError as exc:
            if (
                "out of memory" not in str(exc).lower()
                and "failed to allocate" not in str(exc).lower()
            ):
                raise
            logger.warning(
                "SmolVLM2Stage: device %s ran out of memory — retrying on CPU. "
                "Use --max-images or --torch-device cpu to avoid this.",
                self._model.device,
            )
            self._model = self._model.to("cpu")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self._max_new_tokens,
                )
        inference_ms = (time.perf_counter() - t0) * 1000.0

        t_decode = time.perf_counter()
        new_tokens = generated_ids[:, input_len:]
        decoded = self._processor.batch_decode(new_tokens, skip_special_tokens=True)
        decode_ms = (time.perf_counter() - t_decode) * 1000.0
        caption = self._clean_generation(decoded[0]) if decoded else ""

        if not caption:
            logger.debug("SmolVLM2Stage: empty caption")
            return None

        logger.info(
            "SmolVLM2Stage: generate %.1fms, decode %.1fms, output_chars=%d",
            inference_ms,
            decode_ms,
            len(caption),
        )
        return ClassificationMessage(
            label=caption,
            confidence=1.0,
            all_scores={caption: 1.0},
            timestamp=msg.timestamp,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_generation(decoded: str) -> str:
        """Strip wrapper chat text and keep only the assistant answer."""
        marker = "Assistant:"
        if marker in decoded:
            return decoded.rsplit(marker, maxsplit=1)[-1].strip()
        return decoded.strip()
