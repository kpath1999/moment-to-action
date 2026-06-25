"""Unit tests for LlamaModel and _start_llama_model."""

from __future__ import annotations

import typing as t
from unittest.mock import MagicMock, patch

import pytest

from moment_to_action.hardware._loaded_models._llama import LlamaModel, _start_llama_model
from moment_to_action.hardware._metrics import LlamaCppInferenceMetrics
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType


def _make_llama_model(unit: ComputeUnit = ComputeUnit.CPU) -> LlamaModel:
    """Return a LlamaModel with mock proc and client."""
    return LlamaModel(
        path="/fake/model.gguf",
        mmproj=None,
        port=8080,
        server_path="/usr/bin/llama-server",
        unit=unit,
        proc=MagicMock(),
        client=MagicMock(),
        dtype=DataType.FP32,
    )


@pytest.mark.unit
class TestFindLlamaServer:
    """Tests for __find_llama_server (accessed via _start_llama_model patches)."""

    def test_explicit_path_returned_as_is(self) -> None:
        """When an explicit path is given, it's returned directly."""
        mock_proc = MagicMock()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("httpx.Client", return_value=mock_client),
            patch.object(mock_client, "get", return_value=mock_resp),
            patch(
                "moment_to_action.hardware._loaded_models._llama.pick_free_port", return_value=9999
            ),
        ):
            model = _start_llama_model(
                path="/tmp/model.gguf",
                server_path="/custom/llama-server",
                unit=ComputeUnit.CPU,
                dtype=DataType.FP32,
            )
        assert model._server_path == "/custom/llama-server"

    def test_shutil_which_used_when_no_explicit_path(self) -> None:
        """When server_path is None, shutil.which is consulted."""
        mock_proc = MagicMock()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with (
            patch("shutil.which", return_value="/path/from/which/llama-server"),
            patch("subprocess.Popen", return_value=mock_proc),
            patch("httpx.Client", return_value=mock_client),
            patch.object(mock_client, "get", return_value=mock_resp),
            patch(
                "moment_to_action.hardware._loaded_models._llama.pick_free_port", return_value=9999
            ),
        ):
            model = _start_llama_model(
                path="/tmp/model.gguf",
                server_path=None,
                unit=ComputeUnit.CPU,
                dtype=DataType.FP32,
            )
        assert model._server_path == "/path/from/which/llama-server"

    def test_raises_when_server_not_found(self) -> None:
        """Raises RuntimeError when llama-server is not on PATH and none given."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="llama-server not found"):
                _start_llama_model(
                    path="/tmp/model.gguf", unit=ComputeUnit.CPU, dtype=DataType.FP32
                )


@pytest.mark.unit
class TestStartLlamaModel:
    """Tests for _start_llama_model factory."""

    def test_returns_llama_model(self) -> None:
        """_start_llama_model returns a LlamaModel on success."""
        mock_proc = MagicMock()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("httpx.Client", return_value=mock_client),
            patch.object(mock_client, "get", return_value=mock_resp),
            patch(
                "moment_to_action.hardware._loaded_models._llama.pick_free_port", return_value=9999
            ),
        ):
            model = _start_llama_model(
                path="/tmp/model.gguf",
                server_path="/usr/bin/llama-server",
                unit=ComputeUnit.CPU,
                dtype=DataType.FP32,
            )
        assert isinstance(model, LlamaModel)
        assert model.unit == ComputeUnit.CPU
        assert model._port == 9999

    def test_cpu_only_adds_ngl_zero(self) -> None:
        """cpu_only=True appends --ngl 0 to the server args."""
        mock_proc = MagicMock()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        captured_args: list[list[str]] = []

        def _fake_popen(args: list[str], **_kwargs: object) -> MagicMock:
            captured_args.append(args)
            return mock_proc

        with (
            patch("subprocess.Popen", side_effect=_fake_popen),
            patch("httpx.Client", return_value=mock_client),
            patch.object(mock_client, "get", return_value=mock_resp),
            patch(
                "moment_to_action.hardware._loaded_models._llama.pick_free_port", return_value=9999
            ),
        ):
            _start_llama_model(
                path="/tmp/model.gguf",
                server_path="/usr/bin/llama-server",
                unit=ComputeUnit.CPU,
                cpu_only=True,
                dtype=DataType.FP32,
            )
        assert "--ngl" in captured_args[0]
        assert "0" in captured_args[0]

    def test_mmproj_appended_to_args(self) -> None:
        """Mmproj path is appended with --mmproj flag."""
        mock_proc = MagicMock()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        captured_args: list[list[str]] = []

        def _fake_popen(args: list[str], **_kwargs: object) -> MagicMock:
            captured_args.append(args)
            return mock_proc

        with (
            patch("subprocess.Popen", side_effect=_fake_popen),
            patch("httpx.Client", return_value=mock_client),
            patch.object(mock_client, "get", return_value=mock_resp),
            patch(
                "moment_to_action.hardware._loaded_models._llama.pick_free_port", return_value=9999
            ),
        ):
            _start_llama_model(
                path="/tmp/model.gguf",
                mmproj="/tmp/mmproj.gguf",
                server_path="/usr/bin/llama-server",
                unit=ComputeUnit.CPU,
                dtype=DataType.FP32,
            )
        assert "--mmproj" in captured_args[0]
        assert "/tmp/mmproj.gguf" in captured_args[0]

    def test_timeout_raises_runtime_error(self) -> None:
        """Raises RuntimeError if server doesn't become healthy within timeout."""
        import httpx as _httpx

        mock_proc = MagicMock()
        mock_client = MagicMock()
        mock_client.get.side_effect = _httpx.ConnectError("refused")

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("httpx.Client", return_value=mock_client),
            patch("moment_to_action.hardware._loaded_models._llama._HEALTH_TIMEOUT_S", 0.01),
            patch("moment_to_action.hardware._loaded_models._llama._HEALTH_POLL_S", 0.001),
            patch(
                "moment_to_action.hardware._loaded_models._llama.pick_free_port", return_value=9999
            ),
        ):
            with pytest.raises(RuntimeError, match="did not become healthy"):
                _start_llama_model(
                    path="/tmp/model.gguf",
                    server_path="/usr/bin/llama-server",
                    unit=ComputeUnit.CPU,
                    dtype=DataType.FP32,
                )
        mock_proc.terminate.assert_called_once()
        mock_client.close.assert_called_once()

    def test_uses_pick_free_port_when_port_none(self) -> None:
        """pick_free_port is called when port=None."""
        mock_proc = MagicMock()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("httpx.Client", return_value=mock_client),
            patch.object(mock_client, "get", return_value=mock_resp),
            patch(
                "moment_to_action.hardware._loaded_models._llama.pick_free_port",
                return_value=12345,
            ) as mock_pick,
        ):
            model = _start_llama_model(
                path="/tmp/model.gguf",
                server_path="/usr/bin/llama-server",
                port=None,
                unit=ComputeUnit.CPU,
                dtype=DataType.FP32,
            )
        mock_pick.assert_called_once()
        assert model._port == 12345

    def test_explicit_port_skips_pick_free_port(self) -> None:
        """pick_free_port is NOT called when explicit port is given."""
        mock_proc = MagicMock()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with (
            patch("subprocess.Popen", return_value=mock_proc),
            patch("httpx.Client", return_value=mock_client),
            patch.object(mock_client, "get", return_value=mock_resp),
            patch("moment_to_action.hardware._loaded_models._llama.pick_free_port") as mock_pick,
        ):
            model = _start_llama_model(
                path="/tmp/model.gguf",
                server_path="/usr/bin/llama-server",
                port=8888,
                unit=ComputeUnit.CPU,
                dtype=DataType.FP32,
            )
        mock_pick.assert_not_called()
        assert model._port == 8888


@pytest.mark.unit
class TestLlamaModelProperties:
    """Tests for LlamaModel properties."""

    def test_unit_property(self) -> None:
        """Unit returns the value passed at construction."""
        assert _make_llama_model(ComputeUnit.CPU).unit == ComputeUnit.CPU
        assert _make_llama_model(ComputeUnit.GPU).unit == ComputeUnit.GPU

    def test_dtype_property(self) -> None:
        """Dtype returns the value passed at construction."""
        assert _make_llama_model().dtype == DataType.FP32

    def test_model_type_property(self) -> None:
        """model_type is always LLAMA_CPP."""
        assert _make_llama_model().model_type == ModelType.LLAMA_CPP


@pytest.mark.unit
class TestLlamaModelRun:
    """Tests for LlamaModel.run()."""

    _TIMINGS: t.ClassVar = {
        "prompt_n": 5,
        "prompt_ms": 25.0,
        "prompt_per_token_ms": 5.0,
        "prompt_per_second": 200.0,
        "predicted_n": 10,
        "predicted_ms": 500.0,
        "predicted_per_token_ms": 50.0,
        "predicted_per_second": 20.0,
    }

    def test_run_posts_and_returns_content(self) -> None:
        """run() POSTs to /completion and returns the content field."""
        model = _make_llama_model()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": "hello world"}
        model._client.post.return_value = mock_resp  # type: ignore[union-attr]

        result = model.run({"prompt": "Say hello"})
        assert result == "hello world"
        model._client.post.assert_called_once_with(  # type: ignore[union-attr]
            "/completion", json={"prompt": "Say hello", "stream": False}
        )

    def test_run_captures_timings(self) -> None:
        """run() stores timings in last_inference_metrics when present."""
        model = _make_llama_model()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": "hi", "timings": self._TIMINGS}
        model._client.post.return_value = mock_resp  # type: ignore[union-attr]

        model.run({"prompt": "hi"})
        assert isinstance(model.last_inference_metrics, LlamaCppInferenceMetrics)
        assert model.last_inference_metrics.prompt_n == 5
        assert model.last_inference_metrics.predicted_n == 10

    def test_run_no_timings_leaves_metrics_none(self) -> None:
        """run() does not set last_inference_metrics when timings absent."""
        model = _make_llama_model()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": "hi"}
        model._client.post.return_value = mock_resp  # type: ignore[union-attr]

        model.run({"prompt": "hi"})
        assert model.last_inference_metrics is None

    def test_run_when_unloaded_raises(self) -> None:
        """run() raises RuntimeError after unload()."""
        model = _make_llama_model()
        model.unload()
        with pytest.raises(RuntimeError, match="unloaded"):
            model.run({"prompt": "hi"})

    def test_last_inference_metrics_initially_none(self) -> None:
        """last_inference_metrics is None before any inference."""
        model = _make_llama_model()
        assert model.last_inference_metrics is None


@pytest.mark.unit
class TestLlamaModelStream:
    """Tests for LlamaModel.stream()."""

    def test_stream_yields_content_chunks(self) -> None:
        """stream() yields content from SSE data lines."""
        model = _make_llama_model()

        sse_lines = [
            'data: {"content": "Hello", "stop": false}',
            'data: {"content": " world", "stop": true}',
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        model._client.stream.return_value = mock_resp  # type: ignore[union-attr]

        chunks = list(model.stream({"prompt": "hi", "n_predict": 10}))
        assert chunks == ["Hello", " world"]

    def test_stream_stops_on_done_sentinel(self) -> None:
        """stream() stops at data: [DONE]."""
        model = _make_llama_model()

        sse_lines = [
            'data: {"content": "Token", "stop": false}',
            "data: [DONE]",
            'data: {"content": "Never", "stop": false}',
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        model._client.stream.return_value = mock_resp  # type: ignore[union-attr]

        chunks = list(model.stream({"prompt": "hi"}))
        assert chunks == ["Token"]

    def test_stream_skips_non_data_lines(self) -> None:
        """stream() ignores lines that don't start with 'data: '."""
        model = _make_llama_model()

        sse_lines = [
            ": keep-alive",
            'data: {"content": "Hi", "stop": true}',
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        model._client.stream.return_value = mock_resp  # type: ignore[union-attr]

        chunks = list(model.stream({"prompt": "hi"}))
        assert chunks == ["Hi"]

    def test_stream_when_unloaded_raises(self) -> None:
        """stream() raises RuntimeError after unload()."""
        model = _make_llama_model()
        model.unload()
        with pytest.raises(RuntimeError, match="unloaded"):
            list(model.stream({"prompt": "hi"}))

    def test_stream_captures_timings_from_stop_chunk(self) -> None:
        """stream() stores timings from the stop chunk in last_inference_metrics."""
        model = _make_llama_model()
        timings = {
            "prompt_n": 3,
            "prompt_ms": 15.0,
            "prompt_per_token_ms": 5.0,
            "prompt_per_second": 200.0,
            "predicted_n": 7,
            "predicted_ms": 350.0,
            "predicted_per_token_ms": 50.0,
            "predicted_per_second": 20.0,
        }
        import json as _json

        stop_data = _json.dumps({"content": "", "stop": True, "timings": timings})
        sse_lines = [
            'data: {"content": "Hello", "stop": false}',
            f"data: {stop_data}",
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        model._client.stream.return_value = mock_resp  # type: ignore[union-attr]

        list(model.stream({"prompt": "hi"}))
        assert isinstance(model.last_inference_metrics, LlamaCppInferenceMetrics)
        assert model.last_inference_metrics.prompt_n == 3
        assert model.last_inference_metrics.predicted_n == 7

    def test_stream_no_timings_in_stop_chunk_leaves_metrics_none(self) -> None:
        """stream() does not set last_inference_metrics when stop chunk has no timings."""
        model = _make_llama_model()
        sse_lines = [
            'data: {"content": "Hi", "stop": true}',
        ]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        model._client.stream.return_value = mock_resp  # type: ignore[union-attr]

        list(model.stream({"prompt": "hi"}))
        assert model.last_inference_metrics is None


@pytest.mark.unit
class TestLlamaModelUnload:
    """Tests for LlamaModel.unload()."""

    def test_unload_terminates_proc_and_closes_client(self) -> None:
        """unload() terminates the subprocess and closes the HTTP client."""
        mock_proc = MagicMock()
        mock_client = MagicMock()
        model = LlamaModel(
            path="/fake/model.gguf",
            mmproj=None,
            port=8080,
            server_path="/usr/bin/llama-server",
            unit=ComputeUnit.CPU,
            proc=mock_proc,
            client=mock_client,
            dtype=DataType.FP32,
        )
        model.unload()
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once()
        mock_client.close.assert_called_once()
        assert model._unloaded is True

    def test_unload_idempotent(self) -> None:
        """Calling unload() twice terminates the process only once."""
        mock_proc = MagicMock()
        mock_client = MagicMock()
        model = LlamaModel(
            path="/fake/model.gguf",
            mmproj=None,
            port=8080,
            server_path="/usr/bin/llama-server",
            unit=ComputeUnit.CPU,
            proc=mock_proc,
            client=mock_client,
            dtype=DataType.FP32,
        )
        model.unload()
        model.unload()
        mock_proc.terminate.assert_called_once()
