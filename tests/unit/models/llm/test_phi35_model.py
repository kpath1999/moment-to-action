"""Unit tests for Phi35Model."""

from __future__ import annotations

from pathlib import Path

import pytest

from moment_to_action.hardware._types import ComputeUnit, ModelType
from moment_to_action.models.llm._base import LlamaGGUFModel
from moment_to_action.models.llm.phi35._model import Phi35Model


@pytest.mark.unit
class TestPhi35Model:
    """Tests for Phi35Model."""

    def test_is_subclass_of_llama_gguf_model(self) -> None:
        """Phi35Model is a LlamaGGUFModel subclass."""
        assert issubclass(Phi35Model, LlamaGGUFModel)

    def test_instantiates_without_error(self) -> None:
        """Phi35Model can be instantiated with the expected constructor signature."""
        model = Phi35Model(
            "default",
            Path("/fake/dir"),
            ModelType.LLAMA_CPP,
            backends={ComputeUnit.GPU: {"model": "Phi-3.5-mini-instruct-Q4_0.gguf"}},
            input_layout=None,
            system_prompt="Be concise.",
            max_tokens=128,
        )
        assert model._gguf_path == Path("/fake/dir/Phi-3.5-mini-instruct-Q4_0.gguf")
        assert model._max_tokens == 128
