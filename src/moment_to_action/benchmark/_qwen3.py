from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moment_to_action.benchmark._base import ModelBenchmark
from moment_to_action.models import ModelID

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager


@attrs.frozen
class _Qwen3Handle:
    """Internal handle carrying Qwen3 model and tokenizer."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


class Qwen3Benchmark(ModelBenchmark):
    """Benchmark implementation for Qwen3-4B-Instruct."""

    @property
    def model_id(self) -> ModelID:
        return ModelID.QWEN2_5_4B

    def _load_model(self, backend: ComputeBackend, manager: ModelManager) -> object:
        policy = backend.resolve_torch_policy("auto")
        model_path = manager.get_path(self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=policy.dtype,
        ).to(policy.device)  # type: ignore[arg-type]
        model.train(mode=False)
        return _Qwen3Handle(model=model, tokenizer=tokenizer)

    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object:
        model_handle = self._cast_handle(handle)
        prompts = ["Summarize this frame in one short sentence."] * batch_size
        inputs = model_handle.tokenizer(prompts, return_tensors="pt", padding=True)
        return {name: tensor.to(model_handle.model.device) for name, tensor in inputs.items()}

    def _run_inference(self, handle: object, inputs: object, backend: ComputeBackend) -> None:
        del backend
        model_handle = self._cast_handle(handle)
        if not isinstance(inputs, dict):
            msg = "Qwen3 benchmark expects mapping inputs"
            raise TypeError(msg)
        with torch.inference_mode():
            model_handle.model.generate(  # type: ignore[operator]
                **inputs,
                do_sample=False,
                max_new_tokens=16,
            )

    @staticmethod
    def _cast_handle(handle: object) -> _Qwen3Handle:
        if not isinstance(handle, _Qwen3Handle):
            msg = "Invalid Qwen3 benchmark handle"
            raise TypeError(msg)
        return handle
