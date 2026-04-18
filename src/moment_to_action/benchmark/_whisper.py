from __future__ import annotations

import re
from typing import TYPE_CHECKING

import attrs
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from moment_to_action.benchmark._base import ModelBenchmark
from moment_to_action.models import ModelID

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.processing_utils import ProcessorMixin

    from moment_to_action.benchmark._librispeech_dataset import LibriSpeechDataset
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager


@attrs.frozen
class _WhisperHandle:
    """Internal handle carrying Whisper model and processor."""

    model: PreTrainedModel
    processor: ProcessorMixin


class WhisperTinyBenchmark(ModelBenchmark):
    """Benchmark implementation for Whisper-tiny ASR."""

    def __init__(self, librispeech_dataset: LibriSpeechDataset | None = None) -> None:
        super().__init__()
        self._librispeech_dataset = librispeech_dataset

    @property
    def model_id(self) -> ModelID:
        return ModelID.WHISPER_TINY

    def _load_model(self, backend: ComputeBackend, manager: ModelManager) -> object:
        policy = backend.resolve_torch_policy("auto")
        model_path = manager.get_path(self.model_id)
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=policy.dtype,
        ).to(policy.device)  # type: ignore[arg-type]
        model.train(mode=False)
        return _WhisperHandle(model=model, processor=processor)

    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object:
        model_handle = self._cast_handle(handle)
        del batch_size
        silent_audio = np.zeros((16000,), dtype=np.float32)
        features = model_handle.processor(
            silent_audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        return {
            name: tensor.to(model_handle.model.device)
            for name, tensor in features.items()
            if isinstance(tensor, torch.Tensor)
        }

    def _run_inference(self, handle: object, inputs: object, backend: ComputeBackend) -> None:
        del backend
        model_handle = self._cast_handle(handle)
        if not isinstance(inputs, dict):
            msg = "Whisper benchmark expects mapping inputs"
            raise TypeError(msg)

        with torch.inference_mode():
            model_handle.model.generate(**inputs, max_new_tokens=64)  # type: ignore[operator]

    def _evaluate_accuracy(
        self,
        handle: object,
        backend: ComputeBackend,
        manager: ModelManager,
    ) -> float | None:
        del backend, manager
        dataset = self._librispeech_dataset
        if dataset is None:
            return None

        model_handle = self._cast_handle(handle)
        try:
            import soundfile as sf
            from jiwer import wer
        except ImportError as exc:  # pragma: no cover - guarded by dependency
            msg = "soundfile and jiwer packages are required for WhisperTinyBenchmark accuracy"
            raise RuntimeError(msg) from exc

        predictions: list[str] = []
        references: list[str] = []

        for item in dataset.items():
            audio, sample_rate = sf.read(str(item.audio_path), dtype="float32")
            if isinstance(audio, np.ndarray) and audio.ndim > 1:
                audio = audio.mean(axis=1)

            inputs = model_handle.processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            model_inputs = {
                name: tensor.to(model_handle.model.device)
                for name, tensor in inputs.items()
                if isinstance(tensor, torch.Tensor)
            }

            with torch.inference_mode():
                generated_ids = model_handle.model.generate(  # type: ignore[operator]
                    **model_inputs,
                    max_new_tokens=128,
                )

            decoded = model_handle.processor.batch_decode(generated_ids, skip_special_tokens=True)
            if not decoded:
                continue

            predictions.append(_normalize_text(decoded[0]))
            references.append(_normalize_text(item.transcript))

        if not references:
            return None

        current_wer = float(wer(references, predictions))
        accuracy = max(0.0, 1.0 - current_wer)
        self._set_accuracy_details({"wer": current_wer, "n_evaluated": float(len(references))})
        return accuracy

    @staticmethod
    def _cast_handle(handle: object) -> _WhisperHandle:
        if not isinstance(handle, _WhisperHandle):
            msg = "Invalid Whisper benchmark handle"
            raise TypeError(msg)
        return handle


def _normalize_text(value: str) -> str:
    compact = re.sub(r"\s+", " ", value.lower().strip())
    return re.sub(r"[^a-z0-9 ]", "", compact)
