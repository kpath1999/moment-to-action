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

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from moment_to_action.messages.video import VideoClipMessage
from moment_to_action.messages.vlm import ClassificationMessage
from moment_to_action.metrics._types import SpanType
from moment_to_action.models import ModelID, ModelManager
from moment_to_action.stages._base import Stage
from moment_to_action.utils.video import sample_frames, to_pil_rgb

if TYPE_CHECKING:
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = "Describe the key action happening in these frames."
_DEFAULT_SYSTEM = "Focus on salient action and safety-relevant events. Be concise and factual."

# Maximum number of frames uniformly sampled from a clip per inference call.
# Higher values improve temporal coverage but increase VRAM and latency linearly.
_DEFAULT_MAX_IMAGES = 8

# Maximum number of new tokens the model may generate per call.
# 96 tokens ≈ 2-3 short sentences — enough for a concise scene description.
_DEFAULT_MAX_NEW_TOKENS = 96


class SmolVLM2Stage(Stage):
    """Run SmolVLM2 on sampled video-clip frames and produce a scene description.

    The model is loaded once at construction via HuggingFace Transformers
    and reused across all calls.  Frames are uniformly sampled from the
    incoming :class:`VideoClipMessage`, converted to PIL images, and fed
    through the SmolVLM2 chat template.

    Args:
        backend: Compute backend used to resolve torch policy.
        manager: Model manager used for model resolution and caching.
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
        backend: ComputeBackend,
        manager: ModelManager,
        torch_device: str = "auto",
        prompt: str = _DEFAULT_PROMPT,
        system_prompt: str = _DEFAULT_SYSTEM,
        max_images: int = _DEFAULT_MAX_IMAGES,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._manager = manager
        self._prompt = prompt
        self._system_prompt = system_prompt
        self._max_images = max(1, max_images)
        self._max_new_tokens = max_new_tokens

        policy = self._backend.resolve_torch_policy(torch_device)
        device = policy.device
        dtype = policy.dtype

        model_path = self._manager.get_path(ModelID.SMOLVLM2_2_2B)

        logger.info(
            "SmolVLM2Stage: loading %s (requested=%s, device=%s, dtype=%s)",
            model_path,
            torch_device,
            device,
            dtype,
        )
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
        ).to(device)  # type: ignore[arg-type]  # transformers stubs mis-type .to()
        self._model.train(mode=False)  # put model in inference mode (training=False)
        logger.info("SmolVLM2Stage: ready")

    # ------------------------------------------------------------------
    # Stage interface
    # ------------------------------------------------------------------

    def _process(
        self,
        msg: Message,
        metrics: MetricsCollector,
    ) -> ClassificationMessage | None:
        """Sample frames from a clip, run SmolVLM2, and return a description."""
        if not isinstance(msg, VideoClipMessage):
            type_name = type(msg).__name__
            err = f"SmolVLM2Stage expects VideoClipMessage, got {type_name}"
            raise TypeError(err)

        sampled = sample_frames(msg.frames, self._max_images)
        logger.info(
            "SmolVLM2Stage: sampled %d/%d frames (max_images=%d)",
            len(sampled),
            len(msg.frames),
            self._max_images,
        )

        t_prepare = time.perf_counter()
        images = [to_pil_rgb(f) for f in sampled]

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
        with (
            torch.inference_mode(),
            metrics.start_span(  # Span around model inference
                SpanType.MODEL_INFERENCE, "SmolVLM2 generation"
            ),
        ):
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
