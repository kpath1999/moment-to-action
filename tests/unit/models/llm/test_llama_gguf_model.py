"""Unit tests for LlamaGGUFModel (tested via Qwen2Model)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models._formats import ModelFormat
from moment_to_action.models.llm._base import LlamaGGUFModel
from moment_to_action.models.llm.qwen2._model import Qwen2Model

_BACKENDS: dict[ComputeUnit, dict[str, str]] = {
    ComputeUnit.GPU: {"model": "model.gguf"},
}
_VARIANT_DIR = Path("/fake/variant")
_SYSTEM = "Be concise."


def _make_model(
    system_prompt: str = _SYSTEM,
    max_tokens: int = 64,
) -> Qwen2Model:
    """Construct a Qwen2Model with test parameters."""
    return Qwen2Model(
        "default",
        _VARIANT_DIR,
        ModelFormat.GGUF,
        backends=_BACKENDS,
        input_layout=None,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
    )


@pytest.mark.unit
class TestLlamaGGUFModelConstruction:
    """Tests for LlamaGGUFModel.__init__ via Qwen2Model."""

    def test_gguf_path_resolved_from_backends(self) -> None:
        """_gguf_path joins variant dir with the first backend's filename."""
        model = _make_model()
        assert model._gguf_path == _VARIANT_DIR / "model.gguf"

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
        """Qwen2Model inherits from LlamaGGUFModel."""
        model = _make_model()
        assert isinstance(model, LlamaGGUFModel)


@pytest.mark.unit
class TestLlamaGGUFModelLoad:
    """Tests for LlamaGGUFModel.load()."""

    def test_load_calls_platform_load_llama_cpp(self) -> None:
        """load() delegates to platform.load_llama_cpp with the gguf path."""
        model = _make_model()
        mock_platform = MagicMock()
        mock_loaded = MagicMock()
        mock_platform.load_llama_cpp.return_value = mock_loaded

        model.load(mock_platform, ComputeUnit.CPU)

        mock_platform.load_llama_cpp.assert_called_once_with(
            ComputeUnit.CPU, _VARIANT_DIR / "model.gguf"
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
class TestLlamaGGUFModelUnload:
    """Tests for LlamaGGUFModel.unload()."""

    def _load_model(self, model: Qwen2Model) -> MagicMock:
        """Helper: load model with mock platform."""
        mock_loaded = MagicMock()
        mock_platform = MagicMock()
        mock_platform.load_llama_cpp.return_value = mock_loaded
        model.load(mock_platform, ComputeUnit.CPU)
        return mock_loaded

    def test_unload_calls_loaded_model_unload(self) -> None:
        """unload() calls unload() on the loaded model."""
        model = _make_model()
        mock_loaded = self._load_model(model)

        model.unload()

        mock_loaded.unload.assert_called_once()

    def test_unload_marks_model_not_loaded(self) -> None:
        """After unload(), is_loaded returns False."""
        model = _make_model()
        self._load_model(model)

        model.unload()

        assert not model.is_loaded

    def test_unload_is_idempotent(self) -> None:
        """Second unload() call is a no-op (no error)."""
        model = _make_model()
        self._load_model(model)

        model.unload()
        model.unload()  # must not raise

    def test_unload_when_not_loaded_is_noop(self) -> None:
        """unload() on a never-loaded model does nothing."""
        model = _make_model()
        model.unload()  # must not raise


@pytest.mark.unit
class TestLlamaGGUFModelPrepare:
    """Tests for LlamaGGUFModel.prepare()."""

    def test_prepare_returns_dict(self) -> None:
        """prepare() returns a dict."""
        model = _make_model(system_prompt="sys", max_tokens=32)
        result = model.prepare("hello")
        assert isinstance(result, dict)

    def test_prepare_includes_prompt(self) -> None:
        """prepare() includes system prompt and user prompt in 'prompt' key."""
        model = _make_model(system_prompt="You are helpful.")
        result = model.prepare("what is this?")
        assert "prompt" in result
        assert "You are helpful." in result["prompt"]
        assert "what is this?" in result["prompt"]

    def test_prepare_no_system_prompt(self) -> None:
        """prepare() with empty system_prompt uses the user prompt directly."""
        model = _make_model(system_prompt="")
        result = model.prepare("hello")
        assert result["prompt"] == "hello"

    def test_prepare_n_predict(self) -> None:
        """prepare() sets n_predict from max_tokens."""
        model = _make_model(max_tokens=999)
        result = model.prepare("x")
        assert result["n_predict"] == 999


@pytest.mark.unit
class TestLlamaGGUFModelRun:
    """Tests for LlamaGGUFModel.run()."""

    def test_run_raises_when_not_loaded(self) -> None:
        """run() raises RuntimeError when model is not loaded."""
        model = _make_model()
        with pytest.raises(RuntimeError, match="not loaded"):
            model.run({"prompt": "x", "n_predict": 10})

    def test_run_delegates_to_loaded_model(self) -> None:
        """run() delegates to self._loaded_model.run() and returns result."""
        model = _make_model()
        mock_loaded = MagicMock()
        mock_loaded.run.return_value = "hello world"
        mock_platform = MagicMock()
        mock_platform.load_llama_cpp.return_value = mock_loaded
        model.load(mock_platform, ComputeUnit.CPU)

        result = model.run({"prompt": "x", "n_predict": 10})

        mock_loaded.run.assert_called_once_with({"prompt": "x", "n_predict": 10})
        assert result == "hello world"


@pytest.mark.unit
class TestLlamaGGUFModelPostProc:
    """Tests for LlamaGGUFModel.post_proc()."""

    def test_post_proc_wraps_in_list(self) -> None:
        """post_proc() returns a single-element list."""
        model = _make_model()
        assert model.post_proc("some text") == ["some text"]


@pytest.mark.unit
class TestLlamaGGUFModelVerifyOutputs:
    """Tests for LlamaGGUFModel.verify_outputs()."""

    def test_verify_outputs_raises_not_implemented(self) -> None:
        """verify_outputs() always raises NotImplementedError."""
        model = _make_model()
        with pytest.raises(NotImplementedError):
            model.verify_outputs(None, None, tol=0.1, is_npu=False)
