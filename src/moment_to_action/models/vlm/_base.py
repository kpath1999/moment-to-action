"""Base class for GGUF vision-language models served via llama-server with --mmproj."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from moment_to_action.hardware._loaded_models._llama import LlamaModel
from moment_to_action.metrics import NullMetricsCollector, SpanType
from moment_to_action.models.llm._base import LlamaGGUFModel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from moment_to_action.hardware import Platform
    from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType
    from moment_to_action.metrics import MetricsCollector


class LlamaVLModel(LlamaGGUFModel):
    """Base for multimodal GGUF vision-language models served via llama-server.

    Extends :class:`~moment_to_action.models.llm._base.LlamaGGUFModel` to:

    - resolve a second ``"mmproj"`` artifact (the vision encoder projection weights)
      from the same variant directory as the text GGUF file.
    - pass ``--mmproj <path>`` to the platform's ``load_llama_cpp`` call.
    - accept ``(prompt, images)`` as input to :meth:`prepare`, where ``images`` is a
      list of base64-encoded JPEG strings, and build the multimodal ``/completion``
      request body.

    The three-stage inference pipeline maps to multimodal generation:

    - ``prepare((prompt, b64_images))`` — formats the multimodal ``/completion`` request
    - ``run(prepared)`` — delegates to :class:`~moment_to_action.hardware._loaded_models.LlamaModel`
    - ``post_proc(raw)`` — wraps the text in a list for pipeline compatibility

    Args:
        variant: Registry variant key.
        path: Variant directory containing both the GGUF and mmproj files.
        model_type: File format (``ModelType.LLAMA_CPP``).
        data_type: Quantization type of the model (e.g. ``DataType.FP32``).
        backends: Compute-unit → artifact filename mapping; the first entry
            must contain both a ``"model"`` key (text GGUF) and an ``"mmproj"`` key
            (vision encoder GGUF).
        input_layout: Not applicable to VLMs; expected to be ``None``.
        system_prompt: System message prepended to every completion prompt.
        max_tokens: Maximum tokens the model may generate per call.
    """

    def __init__(
        self,
        variant: str,
        path: Path,
        model_type: ModelType,
        data_type: DataType,
        *,
        backends: dict[ComputeUnit, dict[str, str]],
        input_layout: str | None = None,
        system_prompt: str = "",
        max_tokens: int = 128,
    ) -> None:
        """Initialise with registry metadata and mmproj resolution.

        Args:
            variant: Registry variant key (e.g. ``"default"``).
            path: Variant directory; both the GGUF and mmproj files are resolved
                relative to this path.
            model_type: File format — should be ``ModelType.LLAMA_CPP``.
            data_type: Quantization type of the model (e.g. ``DataType.FP32``).
            backends: Compute-unit → ``{component_name: filename}`` dict.  Must
                contain at least ``"model"`` and ``"mmproj"`` keys in the first entry.
            input_layout: Unused for VLMs; pass ``None``.
            system_prompt: System message prepended to every completion prompt.
            max_tokens: Maximum tokens to generate per completion.
        """
        super().__init__(
            variant,
            path,
            model_type,
            data_type,
            backends=backends,
            input_layout=input_layout,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
        first_unit_backends = next(iter(backends.values()))
        self._mmproj_path = path / first_unit_backends["mmproj"]

    def _load(self, platform: Platform, unit: ComputeUnit) -> None:
        """Load the VLM via the platform's llama-server backend with the mmproj file.

        Args:
            platform: The hardware platform to load on.
            unit: The compute unit to target.

        Raises:
            RuntimeError: If the model is already loaded.
        """
        if self.is_loaded:
            msg = f"{type(self).__name__} is already loaded"
            raise RuntimeError(msg)
        dtype = self._data_type
        self._loaded_model = platform.load_llama_cpp(
            unit, self._gguf_path, mmproj=self._mmproj_path, dtype=dtype
        )
        self._platform = platform

    def _prepare(self, inputs: tuple[str, list[str]]) -> dict:  # type: ignore[override]
        """Format a prompt and base64-encoded images into a multimodal ``/completion`` request.

        Args:
            inputs: ``(prompt, b64_images)`` where ``b64_images`` is a list of
                base64-encoded JPEG strings (without the ``data:`` prefix).

        Returns:
            Request body dict for the llama.cpp ``/completion`` endpoint.
        """
        prompt, b64_images = inputs
        img_tags = "".join(f"[img-{i + 1}]\n" for i in range(len(b64_images)))
        full_prompt = img_tags + (
            f"{self._system_prompt}\n{prompt}" if self._system_prompt else prompt
        )
        return {
            "prompt": full_prompt,
            "image_data": [{"data": b, "id": i + 1} for i, b in enumerate(b64_images)],
            "n_predict": self._max_tokens,
        }

    def stream(  # type: ignore[override]
        self,
        inputs: tuple[str, list[str]],
        *,
        metrics: MetricsCollector | None = None,
    ) -> Generator[str, None, None]:
        """Stream generated tokens for a multimodal prompt.

        Overrides :meth:`~moment_to_action.models.llm._base.LlamaGGUFModel.stream` to accept
        a ``(prompt, b64_images)`` tuple instead of a plain string.  Wraps the stream in a
        ``MODEL_INFERENCE`` span and attaches inference metrics on completion.

        Args:
            inputs: ``(prompt, b64_images)`` where ``b64_images`` is a list of
                base64-encoded JPEG strings.
            metrics: Active collector with an open trace to record the span.

        Yields:
            String token chunks as they arrive from llama-server.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._loaded_model is None:
            msg = f"{type(self).__name__} is not loaded; call load() first"
            raise RuntimeError(msg)
        if metrics is None:
            logger.warning(
                "%s.stream() called without a MetricsCollector;"
                " inference latency will not be recorded",
                type(self).__name__,
            )
            metrics = NullMetricsCollector()
        prepared = self._prepare(inputs)
        if not isinstance(self._loaded_model, LlamaModel):
            msg = f"{type(self).__name__}: streaming requires a LlamaModel loaded model"
            raise TypeError(msg)
        loaded = self._loaded_model
        with metrics.start_span(SpanType.MODEL_INFERENCE, f"{type(self).__name__}.stream") as span:
            yield from loaded.stream(prepared)
            inf_m = loaded.last_inference_metrics
            if inf_m is not None:
                span.inference_metrics = inf_m
