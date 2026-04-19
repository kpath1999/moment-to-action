from __future__ import annotations

import re
from typing import TYPE_CHECKING

import attrs
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from moment_to_action.benchmark._benchmarks._base import ModelBenchmark
from moment_to_action.models import ModelID

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedModel
    from transformers.processing_utils import ProcessorMixin

    from moment_to_action.benchmark._datasets._msrvtt_dataset import MsrvttDataset
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager


@attrs.frozen
class _SmolVLM2Handle:
    """Internal handle carrying SmolVLM2 model and processor."""

    model: PreTrainedModel
    processor: ProcessorMixin


class SmolVLM2Benchmark(ModelBenchmark):
    """Benchmark implementation for SmolVLM2-2.2B."""

    def __init__(self, msrvtt_dataset: MsrvttDataset | None = None) -> None:
        super().__init__()
        self._msrvtt_dataset = msrvtt_dataset

    @property
    def model_id(self) -> ModelID:
        return ModelID.SMOLVLM2_2_2B

    def _load_model(self, backend: ComputeBackend, manager: ModelManager) -> object:
        policy = backend.resolve_torch_policy("auto")
        model_path = manager.get_path(self.model_id)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=policy.dtype,
            trust_remote_code=True,
        ).to(policy.device)  # type: ignore[arg-type]
        model.train(mode=False)
        return _SmolVLM2Handle(model=model, processor=processor)

    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object:
        model_handle = self._cast_handle(handle)
        image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image."},
                    {"type": "image", "image": image},
                ],
            }
        ]
        inputs: dict[str, torch.Tensor] = model_handle.processor.apply_chat_template(  # type: ignore[assignment]
            messages,  # type: ignore[arg-type]
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        del batch_size
        return {name: tensor.to(model_handle.model.device) for name, tensor in inputs.items()}

    def _run_inference(self, handle: object, inputs: object, backend: ComputeBackend) -> None:
        del backend
        model_handle = self._cast_handle(handle)
        if not isinstance(inputs, dict):
            msg = "SmolVLM2 benchmark expects mapping inputs"
            raise TypeError(msg)
        with torch.inference_mode():
            model_handle.model.generate(  # type: ignore[operator]
                **inputs,
                do_sample=False,
                max_new_tokens=8,
            )

    def _evaluate_accuracy(
        self,
        handle: object,
        backend: ComputeBackend,
        manager: ModelManager,
    ) -> float | None:
        del backend, manager
        dataset = self._msrvtt_dataset
        if dataset is None:
            return None

        model_handle = self._cast_handle(handle)
        matches = 0
        evaluated = 0

        for item in dataset.items():
            frames = _sample_video_frames(item.video_path, max_frames=4)
            if not frames:
                continue

            prediction = self._generate_answer(
                model_handle=model_handle,
                question=item.question,
                frames=frames,
            )
            evaluated += 1
            if _normalize_text(prediction) == _normalize_text(item.answer):
                matches += 1

        if evaluated == 0:
            return None

        exact_match = matches / evaluated
        self._set_accuracy_details({"exact_match": exact_match, "n_evaluated": float(evaluated)})
        return exact_match

    @staticmethod
    def _generate_answer(
        model_handle: _SmolVLM2Handle,
        question: str,
        frames: list[Image.Image],
    ) -> str:
        user_content: list[dict[str, object]] = [{"type": "text", "text": question}]
        user_content.extend({"type": "image", "image": frame} for frame in frames)
        messages: list[dict[str, object]] = [
            {"role": "user", "content": user_content},
        ]

        inputs: dict[str, torch.Tensor] = model_handle.processor.apply_chat_template(  # type: ignore[assignment]
            messages,  # type: ignore[arg-type]
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            name: tensor.to(model_handle.model.device) for name, tensor in inputs.items()
        }
        input_len = model_inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generated_ids = model_handle.model.generate(  # type: ignore[operator]
                **model_inputs,
                do_sample=False,
                max_new_tokens=16,
            )

        new_tokens = generated_ids[:, input_len:]
        decoded = model_handle.processor.batch_decode(new_tokens, skip_special_tokens=True)
        if not decoded:
            return ""
        return decoded[0].strip()

    @staticmethod
    def _cast_handle(handle: object) -> _SmolVLM2Handle:
        if not isinstance(handle, _SmolVLM2Handle):
            msg = "Invalid SmolVLM2 benchmark handle"
            raise TypeError(msg)
        return handle


def _sample_video_frames(video_path: Path, max_frames: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames_rgb: list[np.ndarray] = []
    try:
        while len(frames_rgb) < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames_rgb.append(frame_bgr[:, :, ::-1])
    finally:
        cap.release()

    return [Image.fromarray(frame) for frame in frames_rgb]


def _normalize_text(value: str) -> str:
    compact = re.sub(r"\s+", " ", value.lower().strip())
    return re.sub(r"[^a-z0-9 ]", "", compact)
