"""Base class for GGUF language models served via llama-server."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from moment_to_action.hardware._loaded_models._llama import LlamaModel
from moment_to_action.metrics import SpanType
from moment_to_action.models._base import BaseModel
from moment_to_action.prompting import build_payload

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from moment_to_action.hardware import ComputeUnit, DataType, LoadedModel, ModelType, Platform
    from moment_to_action.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class LlamaGGUFModel(BaseModel[str, dict, str, str]):
    """Base for GGUF language models served via llama-server.

    Delegates subprocess management to the hardware layer via
    :meth:`Platform.load_llama_cpp`. Uses the native llama.cpp
    ``/completion`` endpoint.

    The three-stage inference pipeline maps to LLM text generation:

    - ``prepare(prompt)`` — formats the ``/completion`` request body
    - ``run(prepared)`` — delegates to
      :class:`~moment_to_action.hardware._loaded_models.LlamaModel`
    - ``post_proc(raw)`` — wraps the text in a list for pipeline compatibility

    Args:
        variant: Registry variant key.
        path: Variant directory containing the GGUF file.
        model_type: File format (``ModelType.LLAMA_CPP``).
        data_type: Quantization type of the model (e.g. ``DataType.FP32``).
        backends: Compute-unit → artifact filename mapping; the first entry
            must contain a ``"model"`` key naming the ``.gguf`` file.
        input_layout: Not applicable to LLMs; expected to be ``None``.
        system_prompt: System message prepended to every completion prompt.
        max_tokens: Maximum tokens the model may generate per call.
        template: Optional chat template format string with ``{system}``/``{user}``
            placeholders (see :mod:`moment_to_action.prompting`, e.g. ``CHATML``,
            ``PHI3``). ``None`` prepends the system prompt raw (no chat tokens).
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
        template: str | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialise with registry metadata.

        Args:
            variant: Registry variant key (e.g. ``"default"``).
            path: Variant directory; the GGUF file is at
                ``path / next(iter(backends.values()))["model"]``.
            model_type: File format — should be ``ModelType.LLAMA_CPP``.
            data_type: Quantization type of the model (e.g. ``DataType.FP32``).
            backends: Compute-unit → ``{component_name: filename}`` dict.
            input_layout: Unused for LLMs; pass ``None``.
            system_prompt: System message prepended to every completion prompt.
            max_tokens: Maximum tokens to generate per completion.
            template: Optional chat template applied to every prompt (see
                :mod:`moment_to_action.prompting`).
            metrics: Metrics collector used to record ``MODEL_*`` spans.
        """
        super().__init__(
            variant,
            path,
            model_type,
            data_type,
            backends=backends,
            input_layout=input_layout,
            metrics=metrics,
        )
        self._gguf_path = path / next(iter(backends.values()))["model"]
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._template = template
        self._loaded_model: LoadedModel | None = None

    def _load(self, platform: Platform, unit: ComputeUnit) -> None:
        """Load the GGUF model via the platform's llama-server backend.

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
        self._loaded_model = platform.load_llama_cpp(unit, self._gguf_path, dtype=dtype)
        self._platform = platform
        logger.info(
            "%s: loaded %s via platform.load_llama_cpp",
            type(self).__name__,
            self._gguf_path.name,
        )

    def _unload(self) -> None:
        """Unload the model and stop llama-server.

        Safe to call when not loaded (no-op).
        """
        if self._loaded_model is not None:
            self._loaded_model.unload()
            self._loaded_model = None
        self._platform = None

    def _prepare(self, inputs: str, *, grammar: str | None = None) -> dict:
        """Format a user prompt into a ``/completion`` request body.

        Applies :attr:`_template` (if set) via
        :func:`~moment_to_action.prompting.build_payload`; otherwise the system
        prompt is prepended raw.

        Args:
            inputs: User-facing text prompt.
            grammar: Optional GBNF grammar string constraining generation.

        Returns:
            Request body dict for the llama.cpp ``/completion`` endpoint.
        """
        payload = build_payload(inputs, self._max_tokens, self._system_prompt, self._template)
        if grammar is not None:
            payload["grammar"] = grammar
        return payload

    def _run(self, prepared: dict) -> str:
        """Send the completion request and return the generated text.

        Args:
            prepared: Request body from :meth:`prepare`.

        Returns:
            Generated text content.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._loaded_model is None:
            msg = f"{type(self).__name__} is not loaded; call load() first"
            raise RuntimeError(msg)
        return str(self._loaded_model.run(prepared))

    def run(self, prepared: dict) -> str:
        """Run forward pass, recording a ``MODEL_INFERENCE`` span and attaching llama.cpp metrics.

        Overrides :meth:`~moment_to_action.models._base.BaseModel.run` to capture
        :class:`~moment_to_action.hardware.LlamaCppInferenceMetrics` from the
        llama-server response and attach them to the span's ``inference_metrics`` field.

        Args:
            prepared: Output of :meth:`prepare`.

        Returns:
            Generated text content.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        with self._metrics.start_span(
            SpanType.MODEL_INFERENCE, f"{type(self).__name__}.run"
        ) as span:
            result = self._run(prepared)
            if isinstance(self._loaded_model, LlamaModel):
                inf_m = self._loaded_model.last_inference_metrics
                if inf_m is not None:
                    span.inference_metrics = inf_m
        return result

    def stream(self, inputs: str, *, grammar: str | None = None) -> Generator[str, None, None]:
        """Stream generated tokens, recording a ``MODEL_INFERENCE`` span.

        Wraps :meth:`~moment_to_action.hardware._loaded_models._llama.LlamaModel.stream`
        in a ``MODEL_INFERENCE`` metrics span.  The span closes after the generator is
        exhausted or closed early; :class:`~moment_to_action.hardware.LlamaCppInferenceMetrics`
        from the stop chunk are attached to the span's ``inference_metrics`` field when the
        stream runs to completion.

        The span opens on the first ``next()`` call and closes when the generator is
        exhausted or ``.close()``d — including via ``GeneratorExit`` when a downstream
        consumer stops pulling early.

        Args:
            inputs: User prompt string (system prompt is prepended by :meth:`_prepare`).
            grammar: Optional GBNF grammar string constraining generation (e.g. forcing
                a leading ``YES``/``NO`` token). ``None`` preserves unconstrained generation.

        Yields:
            String token chunks as they arrive from llama-server.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._loaded_model is None:
            msg = f"{type(self).__name__} is not loaded; call load() first"
            raise RuntimeError(msg)
        prepared = self._prepare(inputs, grammar=grammar)
        if not isinstance(self._loaded_model, LlamaModel):
            msg = f"{type(self).__name__}: streaming requires a LlamaModel loaded model"
            raise TypeError(msg)
        loaded = self._loaded_model
        with self._metrics.start_span(
            SpanType.MODEL_INFERENCE, f"{type(self).__name__}.stream"
        ) as span:
            try:
                yield from loaded.stream(prepared)
            finally:
                inf_m = loaded.last_inference_metrics
                if inf_m is not None:
                    span.inference_metrics = inf_m

    def _post_proc(self, raw: str) -> list[str]:
        """Wrap the generated text in a list for pipeline compatibility.

        Args:
            raw: Text returned by :meth:`run`.

        Returns:
            Single-element list containing the generated text.
        """
        return [raw]

    def verify_outputs(
        self,
        inputs: object,
        ref_outputs: object,
        *,
        tol: float,
        is_npu: bool,
    ) -> tuple[bool, str]:
        """Not supported — llama-server does not expose per-tensor verification.

        Args:
            inputs: Unused.
            ref_outputs: Unused.
            tol: Unused.
            is_npu: Unused.

        Raises:
            NotImplementedError: Always.
        """
        msg = (
            f"{type(self).__name__} does not support verify_outputs; "
            "llama-server handles inference internally."
        )
        raise NotImplementedError(msg)
