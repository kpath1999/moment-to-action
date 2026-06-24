"""Unit tests for Qwen3VLModel."""

from __future__ import annotations

from pathlib import Path

import pytest

from moment_to_action.hardware._types import ComputeUnit, ModelType
from moment_to_action.models.vlm._base import LlamaVLModel
from moment_to_action.models.vlm.qwen3_vl._model import Qwen3VLModel


@pytest.mark.unit
class TestQwen3VLModel:
    """Tests for Qwen3VLModel."""

    def test_is_subclass_of_llama_vl_model(self) -> None:
        """Qwen3VLModel is a LlamaVLModel subclass."""
        assert issubclass(Qwen3VLModel, LlamaVLModel)

    def test_instantiates_without_error(self) -> None:
        """Qwen3VLModel can be instantiated with the expected constructor signature."""
        model = Qwen3VLModel(
            "default",
            Path("/fake/dir"),
            ModelType.LLAMA_CPP,
            backends={ComputeUnit.GPU: {"model": "model.gguf", "mmproj": "mmproj.gguf"}},
            input_layout=None,
            system_prompt="Be concise.",
            max_tokens=128,
        )
        assert model._mmproj_path == Path("/fake/dir") / "mmproj.gguf"
