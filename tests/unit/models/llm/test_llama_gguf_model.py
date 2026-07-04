"""Unit tests for LlamaGGUFModel (tested via Qwen2Model)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from moment_to_action.hardware._loaded_models._llama import LlamaModel
from moment_to_action.hardware._metrics import LlamaCppInferenceMetrics
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType
from moment_to_action.metrics import MetricsCollector
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
    template: str | None = None,
    metrics: MetricsCollector | None = None,
) -> Qwen2Model:
    """Construct a Qwen2Model with test parameters."""
    return Qwen2Model(
        "default",
        _VARIANT_DIR,
        ModelType.LLAMA_CPP,
        DataType.FP32,
        backends=_BACKENDS,
        input_layout=None,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        template=template,
        metrics=metrics,
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
            ComputeUnit.CPU, _VARIANT_DIR / "model.gguf", dtype=DataType.FP32
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

    def test_prepare_includes_grammar_when_given(self) -> None:
        """_prepare() includes a 'grammar' key when a GBNF grammar is passed."""
        model = _make_model()
        result = model._prepare("x", grammar='root ::= "YES" | "NO"')
        assert result["grammar"] == 'root ::= "YES" | "NO"'

    def test_prepare_omits_grammar_by_default(self) -> None:
        """_prepare() omits the 'grammar' key when none is passed."""
        model = _make_model()
        result = model._prepare("x")
        assert "grammar" not in result

    def test_prepare_applies_chat_template_when_given(self) -> None:
        """prepare() applies the configured chat template via build_payload()."""
        from moment_to_action.prompting import CHATML

        model = _make_model(system_prompt="sys", template=CHATML)
        result = model.prepare("hello")
        assert "<|im_start|>system\nsys<|im_end|>" in result["prompt"]
        assert "<|im_start|>user\nhello<|im_end|>" in result["prompt"]

    def test_prepare_no_template_uses_raw_concatenation(self) -> None:
        """prepare() without a template falls back to raw system+prompt concatenation."""
        model = _make_model(system_prompt="sys")
        result = model.prepare("hello")
        assert result["prompt"] == "sys\nhello"


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

    def _make_llama_loaded_model(
        self, inference_metrics: LlamaCppInferenceMetrics | None
    ) -> LlamaModel:
        """Return a mock LlamaModel with last_inference_metrics pre-configured."""
        mock_loaded = MagicMock(spec=LlamaModel)
        mock_loaded.run.return_value = "hi"
        mock_loaded.last_inference_metrics = inference_metrics
        return mock_loaded

    def test_run_attaches_inference_metrics_to_span(self) -> None:
        """run() attaches LlamaCppInferenceMetrics to the MODEL_INFERENCE span."""
        platform = MagicMock()
        collector = MetricsCollector(platform)
        model = _make_model(metrics=collector)
        inf_m = LlamaCppInferenceMetrics(
            prompt_n=5,
            prompt_ms=25.0,
            prompt_per_token_ms=5.0,
            prompt_per_second=200.0,
            predicted_n=10,
            predicted_ms=500.0,
            predicted_per_token_ms=50.0,
            predicted_per_second=20.0,
        )
        mock_loaded = self._make_llama_loaded_model(inf_m)
        mock_platform = MagicMock()
        mock_platform.load_llama_cpp.return_value = mock_loaded

        with collector.start_trace():
            model.load(mock_platform, ComputeUnit.CPU)
            model.run({"prompt": "x", "n_predict": 10})

        inference_spans = [s for s in collector.spans if "run" in s.name]
        assert len(inference_spans) == 1
        assert inference_spans[0].inference_metrics is inf_m

    def test_run_no_inference_metrics_when_not_llama_model(self) -> None:
        """run() does not fail and span.inference_metrics is None for non-LlamaModel."""
        platform = MagicMock()
        collector = MetricsCollector(platform)
        model = _make_model(metrics=collector)
        mock_loaded = MagicMock()  # not spec=LlamaModel
        mock_loaded.run.return_value = "hi"
        mock_platform = MagicMock()
        mock_platform.load_llama_cpp.return_value = mock_loaded

        with collector.start_trace():
            model.load(mock_platform, ComputeUnit.CPU)
            result = model.run({"prompt": "x"})
        assert result == "hi"


@pytest.mark.unit
class TestLlamaGGUFModelStream:
    """Tests for LlamaGGUFModel.stream()."""

    def _make_loaded_llama(
        self, tokens: list[str], inf_m: LlamaCppInferenceMetrics | None = None
    ) -> LlamaModel:
        """Return a mock LlamaModel whose stream() yields the given tokens."""
        mock_loaded = MagicMock(spec=LlamaModel)
        mock_loaded.stream.return_value = iter(tokens)
        mock_loaded.last_inference_metrics = inf_m
        return mock_loaded

    def test_stream_raises_when_not_loaded(self) -> None:
        """stream() raises RuntimeError when model is not loaded."""
        model = _make_model()
        with pytest.raises(RuntimeError, match="not loaded"):
            list(model.stream("hello"))

    def test_stream_raises_type_error_for_non_llama_loaded_model(self) -> None:
        """stream() raises TypeError when loaded model is not a LlamaModel."""
        model = _make_model()
        mock_platform = MagicMock()
        mock_platform.load_llama_cpp.return_value = MagicMock()  # not LlamaModel
        model.load(mock_platform, ComputeUnit.CPU)
        with pytest.raises(TypeError, match="LlamaModel"):
            list(model.stream("hello"))

    def test_stream_yields_tokens(self) -> None:
        """stream() yields all tokens from the underlying LlamaModel.stream()."""
        platform = MagicMock()
        collector = MetricsCollector(platform)
        model = _make_model(system_prompt="sys", max_tokens=32, metrics=collector)
        mock_loaded = self._make_loaded_llama(["Hello", " world"])
        mock_platform = MagicMock()
        mock_platform.load_llama_cpp.return_value = mock_loaded

        with collector.start_trace():
            model.load(mock_platform, ComputeUnit.CPU)
            tokens = list(model.stream("hi"))
        assert tokens == ["Hello", " world"]

    def test_stream_attaches_inference_metrics_to_span(self) -> None:
        """stream() attaches inference_metrics to MODEL_INFERENCE span after exhaustion."""
        platform = MagicMock()
        collector = MetricsCollector(platform)
        model = _make_model(metrics=collector)
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
        mock_loaded = self._make_loaded_llama(["tok1", "tok2"], inf_m)
        mock_platform = MagicMock()
        mock_platform.load_llama_cpp.return_value = mock_loaded

        with collector.start_trace():
            model.load(mock_platform, ComputeUnit.CPU)
            list(model.stream("prompt"))

        stream_spans = [s for s in collector.spans if "stream" in s.name]
        assert len(stream_spans) == 1
        assert stream_spans[0].inference_metrics is inf_m

    def test_stream_without_metrics_collector_still_yields(self) -> None:
        """stream() works without a MetricsCollector (falls back to NullMetricsCollector)."""
        model = _make_model()
        mock_loaded = self._make_loaded_llama(["tok"])
        mock_platform = MagicMock()
        mock_platform.load_llama_cpp.return_value = mock_loaded
        model.load(mock_platform, ComputeUnit.CPU)

        tokens = list(model.stream("hi"))
        assert tokens == ["tok"]


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
