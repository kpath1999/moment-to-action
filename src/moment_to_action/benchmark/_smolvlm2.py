from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from moment_to_action.benchmark._base import ModelBenchmark
from moment_to_action.models import ModelID

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.processing_utils import ProcessorMixin

    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager


@attrs.frozen
class _SmolVLM2Handle:
    """Internal handle carrying SmolVLM2 model and processor."""

    model: PreTrainedModel
    processor: ProcessorMixin


class SmolVLM2Benchmark(ModelBenchmark):
    """Benchmark implementation for SmolVLM2-2.2B."""

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

    @staticmethod
    def _cast_handle(handle: object) -> _SmolVLM2Handle:
        if not isinstance(handle, _SmolVLM2Handle):
            msg = "Invalid SmolVLM2 benchmark handle"
            raise TypeError(msg)
        return handle
