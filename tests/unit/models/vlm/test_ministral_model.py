"""Unit tests for MinistralModel."""

from __future__ import annotations

from pathlib import Path

import pytest

from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType
from moment_to_action.models.vlm._base import LlamaVLModel
from moment_to_action.models.vlm.ministral._model import MinistralModel


@pytest.mark.unit
class TestMinistralModel:
    """Tests for MinistralModel."""

    def test_is_subclass_of_llama_vl_model(self) -> None:
        """MinistralModel is a LlamaVLModel subclass."""
        assert issubclass(MinistralModel, LlamaVLModel)

    def test_instantiates_without_error(self) -> None:
        """MinistralModel can be instantiated with the expected constructor signature."""
        model = MinistralModel(
            "default",
            Path("/fake/dir"),
            ModelType.LLAMA_CPP,
            DataType.FP32,
            backends={ComputeUnit.GPU: {"model": "model.gguf", "mmproj": "mmproj.gguf"}},
            input_layout=None,
            system_prompt="Be concise.",
            max_tokens=128,
        )
        assert model._mmproj_path == Path("/fake/dir") / "mmproj.gguf"
