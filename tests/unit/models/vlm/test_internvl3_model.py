"""Unit tests for InternVL3Model."""

from __future__ import annotations

from pathlib import Path

import pytest

from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType
from moment_to_action.models.vlm._base import LlamaVLModel
from moment_to_action.models.vlm.internvl3._model import InternVL3Model


@pytest.mark.unit
class TestInternVL3Model:
    """Tests for InternVL3Model."""

    def test_is_subclass_of_llama_vl_model(self) -> None:
        """InternVL3Model is a LlamaVLModel subclass."""
        assert issubclass(InternVL3Model, LlamaVLModel)

    def test_instantiates_without_error(self) -> None:
        """InternVL3Model can be instantiated with the expected constructor signature."""
        model = InternVL3Model(
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
