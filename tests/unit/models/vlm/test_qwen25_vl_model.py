"""Unit tests for Qwen25VLModel."""

from __future__ import annotations

from pathlib import Path

import pytest

from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models._formats import ModelFormat
from moment_to_action.models.vlm._base import LlamaVLModel
from moment_to_action.models.vlm.qwen25_vl._model import Qwen25VLModel


@pytest.mark.unit
class TestQwen25VLModel:
    """Tests for Qwen25VLModel."""

    def test_is_subclass_of_llama_vl_model(self) -> None:
        """Qwen25VLModel is a LlamaVLModel subclass."""
        assert issubclass(Qwen25VLModel, LlamaVLModel)

    def test_instantiates_without_error(self) -> None:
        """Qwen25VLModel can be instantiated with the expected constructor signature."""
        model = Qwen25VLModel(
            "default",
            Path("/fake/dir"),
            ModelFormat.GGUF,
            backends={ComputeUnit.GPU: {"model": "model.gguf", "mmproj": "mmproj.gguf"}},
            input_layout=None,
            server_path=Path("/usr/bin/llama-server"),
            port=8080,
            system_prompt="Be concise.",
            max_tokens=128,
        )
        assert model._port == 8080
        assert model._mmproj_path == Path("/fake/dir") / "mmproj.gguf"
