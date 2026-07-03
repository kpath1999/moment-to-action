"""Unit tests for SmolVLM2Model."""

from __future__ import annotations

from pathlib import Path

import pytest

from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType
from moment_to_action.models.vlm._base import LlamaVLModel
from moment_to_action.models.vlm.smolvlm2._model import SmolVLM2Model


@pytest.mark.unit
class TestSmolVLM2Model:
    """Tests for SmolVLM2Model."""

    def test_is_subclass_of_llama_vl_model(self) -> None:
        """SmolVLM2Model is a LlamaVLModel subclass."""
        assert issubclass(SmolVLM2Model, LlamaVLModel)

    def test_instantiates_without_error(self) -> None:
        """SmolVLM2Model can be instantiated with the expected constructor signature."""
        model = SmolVLM2Model(
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
