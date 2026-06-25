"""Unit tests for LlamaVLModel (tested via Qwen25VLModel)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from moment_to_action.hardware._loaded_models._llama import LlamaModel
from moment_to_action.hardware._metrics import LlamaCppInferenceMetrics
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models.llm._base import LlamaGGUFModel
from moment_to_action.models.vlm._base import LlamaVLModel
from moment_to_action.models.vlm.qwen25_vl._model import Qwen25VLModel

_BACKENDS: dict[ComputeUnit, dict[str, str]] = {
    ComputeUnit.GPU: {
        "model": "model.gguf",
        "mmproj": "mmproj.gguf",
    },
}
_VARIANT_DIR = Path("/fake/variant")
_SYSTEM = "You are a vision AI."


def _make_model(
    system_prompt: str = _SYSTEM,
    max_tokens: int = 64,
) -> Qwen25VLModel:
    """Construct a Qwen25VLModel with test parameters."""
    return Qwen25VLModel(
        "default",
        _VARIANT_DIR,
        ModelType.LLAMA_CPP,
        DataType.FP32,
        backends=_BACKENDS,
        input_layout=None,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
    )


@pytest.mark.unit
class TestLlamaVLModelConstruction:
    """Tests for LlamaVLModel.__init__ via Qwen25VLModel."""

    def test_gguf_path_resolved_from_backends(self) -> None:
        """_gguf_path joins variant dir with model filename."""
        model = _make_model()
        assert model._gguf_path == _VARIANT_DIR / "model.gguf"

    def test_mmproj_path_resolved_from_backends(self) -> None:
        """_mmproj_path joins variant dir with mmproj filename."""
        model = _make_model()
        assert model._mmproj_path == _VARIANT_DIR / "mmproj.gguf"

    def test_system_prompt_stored(self) -> None:
        """_system_prompt is stored as provided."""
        model = _make_model(system_prompt="Custom.")
        assert model._system_prompt == "Custom."

    def test_max_tokens_stored(self) -> None:
        """_max_tokens is stored as provided."""
        model = _make_model(max_tokens=256)
        assert model._max_tokens == 256

    def test_initially_not_loaded(self) -> None:
        """Model is not loaded after construction."""
        model = _make_model()
        assert not model.is_loaded

    def test_is_subclass_of_llama_gguf_model(self) -> None:
        """LlamaVLModel inherits from LlamaGGUFModel."""
        model = _make_model()
        assert isinstance(model, LlamaGGUFModel)

    def test_is_subclass_of_llama_vl_model(self) -> None:
        """Qwen25VLModel inherits from LlamaVLModel."""
        model = _make_model()
        assert isinstance(model, LlamaVLModel)


@pytest.mark.unit
class TestLlamaVLModelLoad:
    """Tests for LlamaVLModel.load()."""

    def test_load_calls_platform_load_llama_cpp_with_mmproj(self) -> None:
        """load() delegates to platform.load_llama_cpp with mmproj kwarg."""
        model = _make_model()
        mock_platform = MagicMock()
        mock_loaded = MagicMock()
        mock_platform.load_llama_cpp.return_value = mock_loaded

        model.load(mock_platform, ComputeUnit.CPU)

        mock_platform.load_llama_cpp.assert_called_once_with(
            ComputeUnit.CPU,
            _VARIANT_DIR / "model.gguf",
            mmproj=_VARIANT_DIR / "mmproj.gguf",
            dtype=DataType.FP32,
        )

    def test_load_marks_model_as_loaded(self) -> None:
        """After load(), is_loaded returns True."""
        model = _make_model()
        mock_platform = MagicMock()
        mock_platform.load_llama_cpp.return_value = MagicMock()

        model.load(mock_platform, ComputeUnit.CPU)

        assert model.is_loaded

    def test_load_raises_if_already_loaded(self) -> None:
        """load() raises RuntimeError when called on an already-loaded model."""
        model = _make_model()
        mock_platform = MagicMock()
        mock_platform.load_llama_cpp.return_value = MagicMock()

        model.load(mock_platform, ComputeUnit.CPU)
        with pytest.raises(RuntimeError, match="already loaded"):
            model.load(mock_platform, ComputeUnit.CPU)


@pytest.mark.unit
class TestLlamaVLModelPrepare:
    """Tests for LlamaVLModel.prepare()."""

    def test_prepare_returns_dict(self) -> None:
        """prepare() returns a dict."""
        model = _make_model()
        result = model.prepare(("describe this", ["abc123"]))  # type: ignore[arg-type]
        assert isinstance(result, dict)

    def test_prepare_includes_prompt_key(self) -> None:
        """prepare() produces a 'prompt' key with img tags and text."""
        model = _make_model(system_prompt="")
        result = model.prepare(("describe this", ["abc123"]))  # type: ignore[arg-type]
        assert "prompt" in result
        assert "describe this" in result["prompt"]

    def test_prepare_includes_image_data(self) -> None:
        """prepare() includes image_data entries for each base64 image."""
        model = _make_model()
        result = model.prepare(("x", ["b64a", "b64b"]))  # type: ignore[arg-type]
        img_data = result["image_data"]
        assert len(img_data) == 2
        assert img_data[0] == {"data": "b64a", "id": 1}
        assert img_data[1] == {"data": "b64b", "id": 2}

    def test_prepare_n_predict(self) -> None:
        """prepare() sets n_predict from max_tokens."""
        model = _make_model(max_tokens=999)
        result = model.prepare(("x", []))  # type: ignore[arg-type]
        assert result["n_predict"] == 999

    def test_prepare_img_tags_in_prompt(self) -> None:
        """prepare() includes [img-N] tags in the prompt for each image."""
        model = _make_model(system_prompt="")
        result = model.prepare(("describe", ["img1", "img2"]))  # type: ignore[arg-type]
        prompt = result["prompt"]
        assert "[img-1]" in prompt
        assert "[img-2]" in prompt

    def test_prepare_no_images(self) -> None:
        """prepare() handles an empty image list."""
        model = _make_model(system_prompt="")
        result = model.prepare(("text only", []))  # type: ignore[arg-type]
        assert result["image_data"] == []
        assert "text only" in result["prompt"]


@pytest.mark.unit
class TestLlamaVLModelStream:
    """Tests for LlamaVLModel.stream()."""

    def _load_with_mock_llama(
        self, model: Qwen25VLModel, tokens: list[str], inf_m: LlamaCppInferenceMetrics | None = None
    ) -> LlamaModel:
        """Load model with a mock LlamaModel whose stream() yields given tokens."""
        mock_loaded = MagicMock(spec=LlamaModel)
        mock_loaded.stream.return_value = iter(tokens)
        mock_loaded.last_inference_metrics = inf_m
        mock_platform = MagicMock()
        mock_platform.load_llama_cpp.return_value = mock_loaded
        model.load(mock_platform, ComputeUnit.CPU)
        return mock_loaded

    def test_stream_raises_when_not_loaded(self) -> None:
        """stream() raises RuntimeError when model is not loaded."""
        model = _make_model()
        with pytest.raises(RuntimeError, match="not loaded"):
            list(model.stream(("hi", [])))

    def test_stream_yields_tokens(self) -> None:
        """stream() yields all tokens from the underlying LlamaModel.stream()."""
        model = _make_model(system_prompt="", max_tokens=32)
        self._load_with_mock_llama(model, ["Hello", " world"])

        platform = MagicMock()
        collector = MetricsCollector(platform)
        with collector.start_trace():
            tokens = list(model.stream(("describe", ["b64img"]), metrics=collector))
        assert tokens == ["Hello", " world"]

    def test_stream_attaches_inference_metrics_to_span(self) -> None:
        """stream() attaches inference_metrics to MODEL_INFERENCE span after exhaustion."""
        model = _make_model()
        inf_m = LlamaCppInferenceMetrics(
            prompt_n=3,
            prompt_ms=15.0,
            prompt_per_token_ms=5.0,
            prompt_per_second=200.0,
            predicted_n=7,
            predicted_ms=350.0,
            predicted_per_token_ms=50.0,
            predicted_per_second=20.0,
        )
        self._load_with_mock_llama(model, ["tok1"], inf_m)

        platform = MagicMock()
        collector = MetricsCollector(platform)
        with collector.start_trace():
            list(model.stream(("prompt", ["b64"]), metrics=collector))

        stream_spans = [s for s in collector.spans if "stream" in s.name]
        assert len(stream_spans) == 1
        assert stream_spans[0].inference_metrics is inf_m

    def test_stream_raises_type_error_for_non_llama_loaded_model(self) -> None:
        """stream() raises TypeError when loaded model is not a LlamaModel."""
        model = _make_model()
        mock_platform = MagicMock()
        mock_platform.load_llama_cpp.return_value = MagicMock()  # not LlamaModel
        model.load(mock_platform, ComputeUnit.CPU)
        with pytest.raises(TypeError, match="LlamaModel"):
            list(model.stream(("hi", [])))
