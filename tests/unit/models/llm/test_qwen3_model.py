"""Unit tests for Qwen3Model."""

from __future__ import annotations

from pathlib import Path

import pytest

from moment_to_action.hardware._types import ComputeUnit, ModelType
from moment_to_action.models.llm._base import LlamaGGUFModel
from moment_to_action.models.llm.qwen3._model import Qwen3Model


@pytest.mark.unit
class TestQwen3Model:
    """Tests for Qwen3Model."""

    def test_is_subclass_of_llama_gguf_model(self) -> None:
        """Qwen3Model is a LlamaGGUFModel subclass."""
        assert issubclass(Qwen3Model, LlamaGGUFModel)

    def test_instantiates_without_error(self) -> None:
        """Qwen3Model can be instantiated with the expected constructor signature."""
        model = Qwen3Model(
            "default",
            Path("/fake/dir"),
            ModelType.LLAMA_CPP,
            backends={ComputeUnit.GPU: {"model": "Qwen3-4B-Q4_K_M.gguf"}},
            input_layout=None,
            system_prompt="Be concise.",
            max_tokens=128,
        )
        assert model._gguf_path == Path("/fake/dir/Qwen3-4B-Q4_K_M.gguf")
        assert model._max_tokens == 128
