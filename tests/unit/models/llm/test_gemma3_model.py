"""Unit tests for Gemma3Model."""

from __future__ import annotations

from pathlib import Path

import pytest

from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType
from moment_to_action.models.llm._base import LlamaGGUFModel
from moment_to_action.models.llm.gemma3._model import Gemma3Model


@pytest.mark.unit
class TestGemma3Model:
    """Tests for Gemma3Model."""

    def test_is_subclass_of_llama_gguf_model(self) -> None:
        """Gemma3Model is a LlamaGGUFModel subclass."""
        assert issubclass(Gemma3Model, LlamaGGUFModel)

    def test_instantiates_without_error(self) -> None:
        """Gemma3Model can be instantiated with the expected constructor signature."""
        model = Gemma3Model(
            "default",
            Path("/fake/dir"),
            ModelType.LLAMA_CPP,
            DataType.FP32,
            backends={ComputeUnit.GPU: {"model": "gemma-3-1b-it-Q4_K_M.gguf"}},
            input_layout=None,
            system_prompt="Be concise.",
            max_tokens=128,
        )
        assert model._gguf_path == Path("/fake/dir/gemma-3-1b-it-Q4_K_M.gguf")
        assert model._max_tokens == 128
