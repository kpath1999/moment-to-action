"""Unit tests for Qwen2Model."""

from __future__ import annotations

from pathlib import Path

import pytest

from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models._formats import ModelFormat
from moment_to_action.models.llm._base import LlamaGGUFModel
from moment_to_action.models.llm.qwen2._model import Qwen2Model


@pytest.mark.unit
class TestQwen2Model:
    """Tests for Qwen2Model."""

    def test_is_subclass_of_llama_gguf_model(self) -> None:
        """Qwen2Model is a LlamaGGUFModel subclass."""
        assert issubclass(Qwen2Model, LlamaGGUFModel)

    def test_instantiates_without_error(self) -> None:
        """Qwen2Model can be instantiated with the expected constructor signature."""
        model = Qwen2Model(
            "default",
            Path("/fake/dir"),
            ModelFormat.GGUF,
            backends={ComputeUnit.GPU: {"model": "model.gguf"}},
            input_layout=None,
            system_prompt="Be concise.",
            max_tokens=128,
        )
        assert model._max_tokens == 128
