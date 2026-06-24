"""Base class for GGUF language models served via llama-server."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from moment_to_action.models._base import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

    from moment_to_action.hardware import Platform
    from moment_to_action.hardware._loaded_model import LoadedModel
    from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

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
    """

    def __init__(
        self,
        variant: str,
        path: Path,
        model_type: ModelType | None = None,
        data_type: DataType | None = None,
        *,
        backends: dict[ComputeUnit, dict[str, str]],
        input_layout: str | None = None,
        system_prompt: str = "",
        max_tokens: int = 128,
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
        """
        super().__init__(
            variant,
            path,
            model_type,
            data_type,
            backends=backends,
            input_layout=input_layout,
        )
        self._gguf_path = path / next(iter(backends.values()))["model"]
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._loaded_model: LoadedModel | None = None

    def load(self, platform: Platform, unit: ComputeUnit) -> None:
        """Load the GGUF model via the platform's llama-server backend.

        Args:
            platform: The hardware platform to load on.
            unit: The compute unit to target.

        Raises:
            RuntimeError: If the model is already loaded.
            RuntimeError: If ``data_type`` was not set in the registry entry.
        """
        if self.is_loaded:
            msg = f"{type(self).__name__} is already loaded"
            raise RuntimeError(msg)
        dtype = self._data_type
        if dtype is None:
            msg = "data_type is required for llama.cpp models; check registry entry"
            raise RuntimeError(msg)
        self._loaded_model = platform.load_llama_cpp(unit, self._gguf_path, dtype=dtype)
        self._platform = platform
        logger.info(
            "%s: loaded %s via platform.load_llama_cpp",
            type(self).__name__,
            self._gguf_path.name,
        )

    def unload(self) -> None:
        """Unload the model and stop llama-server.

        Safe to call when not loaded (no-op).
        """
        if self._loaded_model is not None:
            self._loaded_model.unload()
            self._loaded_model = None
        self._platform = None

    def prepare(self, inputs: str) -> dict:
        """Format a user prompt into a ``/completion`` request body.

        Args:
            inputs: User-facing text prompt.

        Returns:
            Request body dict for the llama.cpp ``/completion`` endpoint.
        """
        prompt = f"{self._system_prompt}\n{inputs}" if self._system_prompt else inputs
        return {"prompt": prompt, "n_predict": self._max_tokens}

    def run(self, prepared: dict) -> str:
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

    def post_proc(self, raw: str) -> list[str]:
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
