from __future__ import annotations

import re
from typing import TYPE_CHECKING

import attrs
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moment_to_action.benchmark._benchmarks._base import ModelBenchmark
from moment_to_action.models import ModelID

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from moment_to_action.benchmark._datasets._gsm8k_dataset import GSM8KDataset
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager


@attrs.frozen
class _Qwen3Handle:
    """Internal handle carrying Qwen3 model and tokenizer."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


class Qwen3Benchmark(ModelBenchmark):
    """Benchmark implementation for Qwen3-4B-Instruct."""

    def __init__(self, gsm8k_dataset: GSM8KDataset | None = None) -> None:
        super().__init__()
        self._gsm8k_dataset = gsm8k_dataset

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

    def _evaluate_accuracy(
        self,
        handle: object,
        backend: ComputeBackend,
        manager: ModelManager,
    ) -> float | None:
        del backend, manager
        dataset = self._gsm8k_dataset
        if dataset is None:
            return None

        model_handle = self._cast_handle(handle)
        matches = 0
        evaluated = 0

        for item in dataset.items():
            prediction = self._generate_answer(model_handle=model_handle, prompt=item.question)
            predicted_number = _extract_numeric_answer(prediction)
            if predicted_number is None:
                continue

            evaluated += 1
            if predicted_number == item.answer:
                matches += 1

        if evaluated == 0:
            return None

        exact_match = matches / evaluated
        self._set_accuracy_details({"exact_match": exact_match, "n_evaluated": float(evaluated)})
        return exact_match

    @staticmethod
    def _generate_answer(model_handle: _Qwen3Handle, prompt: str) -> str:
        inputs = model_handle.tokenizer([prompt], return_tensors="pt", padding=True)
        model_inputs = {
            name: tensor.to(model_handle.model.device) for name, tensor in inputs.items()
        }
        input_len = model_inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated_ids = model_handle.model.generate(  # type: ignore[operator]
                **model_inputs,
                do_sample=False,
                max_new_tokens=128,
            )

        new_tokens = generated_ids[:, input_len:]
        decoded = model_handle.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        if not decoded:
            return ""
        return decoded[0].strip()

    @staticmethod
    def _cast_handle(handle: object) -> _Qwen3Handle:
        if not isinstance(handle, _Qwen3Handle):
            msg = "Invalid Qwen3 benchmark handle"
            raise TypeError(msg)
        return handle


_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def _extract_numeric_answer(text: str) -> str | None:
    segment = text.rsplit("####", maxsplit=1)[-1] if "####" in text else text

    matches = _NUMBER_RE.findall(segment)
    if not matches and segment is not text:
        matches = _NUMBER_RE.findall(text)
    if not matches:
        return None

    compact = matches[-1].replace(",", "")
    try:
        value = float(compact)
    except ValueError:
        return compact

    if value.is_integer():
        return str(int(value))
    return str(value)
