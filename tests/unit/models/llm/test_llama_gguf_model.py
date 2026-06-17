"""Unit tests for LlamaGGUFModel (tested via Qwen2Model)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models._formats import ModelFormat
from moment_to_action.models.llm._base import LlamaGGUFModel, _wait_for_server
from moment_to_action.models.llm.qwen2._model import Qwen2Model

_BACKENDS: dict[ComputeUnit, dict[str, str]] = {
    ComputeUnit.CPU: {"model": "model.gguf"},
}
_VARIANT_DIR = Path("/fake/variant")
_SERVER_PATH = Path("/usr/bin/llama-server")
_SYSTEM = "Be concise."


def _make_model(
    port: int = 8080,
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
        server_path=_SERVER_PATH,
        port=port,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
    )


@pytest.mark.unit
class TestLlamaGGUFModelConstruction:
    """Tests for LlamaGGUFModel.__init__ via Qwen2Model."""

    def test_gguf_path_resolved_from_backends(self) -> None:
        """_gguf_path joins variant dir with CPU backend filename."""
        model = _make_model()
        assert model._gguf_path == _VARIANT_DIR / "model.gguf"

    def test_server_path_stored(self) -> None:
        """_server_path is stored as provided."""
        model = _make_model()
        assert model._server_path == _SERVER_PATH

    def test_port_stored(self) -> None:
        """_port is stored as provided."""
        model = _make_model(port=9999)
        assert model._port == 9999

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

    def test_load_starts_subprocess_with_correct_args(self) -> None:
        """load() launches llama-server with the right CLI arguments."""
        model = _make_model(port=8080)
        mock_backend = MagicMock()

        with (
            patch("moment_to_action.models.llm._base.subprocess.Popen") as mock_popen,
            patch("moment_to_action.models.llm._base._wait_for_server"),
        ):
            mock_proc = MagicMock()
            mock_proc.pid = 12345
            mock_popen.return_value = mock_proc

            model.load(mock_backend)

        mock_popen.assert_called_once_with(
            [
                str(_SERVER_PATH),
                "-m",
                str(_VARIANT_DIR / "model.gguf"),
                "--port",
                "8080",
                "--host",
                "127.0.0.1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def test_load_creates_httpx_client(self) -> None:
        """load() creates an httpx.Client pointed at localhost:port."""
        model = _make_model(port=7777)
        mock_backend = MagicMock()

        with (
            patch("moment_to_action.models.llm._base.subprocess.Popen"),
            patch("moment_to_action.models.llm._base._wait_for_server"),
            patch("moment_to_action.models.llm._base.httpx.Client") as mock_client_cls,
        ):
            model.load(mock_backend)

        mock_client_cls.assert_called_once_with(base_url="http://127.0.0.1:7777")

    def test_load_calls_wait_for_server(self) -> None:
        """load() calls _wait_for_server with the httpx client."""
        model = _make_model()
        mock_backend = MagicMock()

        with (
            patch("moment_to_action.models.llm._base.subprocess.Popen"),
            patch("moment_to_action.models.llm._base.httpx.Client") as mock_client_cls,
            patch("moment_to_action.models.llm._base._wait_for_server") as mock_wait,
        ):
            mock_client_instance = MagicMock()
            mock_client_cls.return_value = mock_client_instance
            model.load(mock_backend)

        mock_wait.assert_called_once_with(mock_client_instance)

    def test_load_marks_model_as_loaded(self) -> None:
        """After load(), is_loaded returns True."""
        model = _make_model()

        with (
            patch("moment_to_action.models.llm._base.subprocess.Popen"),
            patch("moment_to_action.models.llm._base._wait_for_server"),
        ):
            model.load(MagicMock())

        assert model.is_loaded

    def test_load_raises_if_already_loaded(self) -> None:
        """load() raises RuntimeError when called on an already-loaded model."""
        model = _make_model()

        with (
            patch("moment_to_action.models.llm._base.subprocess.Popen"),
            patch("moment_to_action.models.llm._base._wait_for_server"),
        ):
            model.load(MagicMock())
            with pytest.raises(RuntimeError, match="already loaded"):
                model.load(MagicMock())


@pytest.mark.unit
class TestLlamaGGUFModelUnload:
    """Tests for LlamaGGUFModel.unload()."""

    def _load_model(self, model: Qwen2Model) -> MagicMock:
        """Helper: load model with mocked subprocess and client."""
        mock_proc = MagicMock()
        mock_proc.pid = 1
        with (
            patch(
                "moment_to_action.models.llm._base.subprocess.Popen",
                return_value=mock_proc,
            ),
            patch("moment_to_action.models.llm._base._wait_for_server"),
        ):
            model.load(MagicMock())
        return mock_proc

    def test_unload_terminates_subprocess(self) -> None:
        """unload() calls terminate() and wait() on the subprocess."""
        model = _make_model()
        mock_proc = self._load_model(model)

        model.unload()

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once()

    def test_unload_closes_client(self) -> None:
        """unload() closes the httpx client."""
        model = _make_model()

        mock_client = MagicMock()
        with (
            patch("moment_to_action.models.llm._base.subprocess.Popen"),
            patch("moment_to_action.models.llm._base._wait_for_server"),
            patch("moment_to_action.models.llm._base.httpx.Client", return_value=mock_client),
        ):
            model.load(MagicMock())

        model.unload()
        mock_client.close.assert_called_once()

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

    def test_prepare_system_message(self) -> None:
        """prepare() includes the system prompt as the first message."""
        model = _make_model(system_prompt="You are helpful.")
        result = model.prepare("hello")
        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}

    def test_prepare_user_message(self) -> None:
        """prepare() includes the user prompt as the second message."""
        model = _make_model()
        result = model.prepare("what is this?")
        assert result["messages"][1] == {"role": "user", "content": "what is this?"}

    def test_prepare_max_tokens(self) -> None:
        """prepare() includes max_tokens from constructor."""
        model = _make_model(max_tokens=999)
        result = model.prepare("x")
        assert result["max_tokens"] == 999


@pytest.mark.unit
class TestLlamaGGUFModelRun:
    """Tests for LlamaGGUFModel.run()."""

    def test_run_raises_when_not_loaded(self) -> None:
        """run() raises RuntimeError when client is None."""
        model = _make_model()
        with pytest.raises(RuntimeError, match="not loaded"):
            model.run({"messages": [], "max_tokens": 10})

    def test_run_posts_to_chat_completions(self) -> None:
        """run() POSTs to /v1/chat/completions with the prepared body."""
        model = _make_model()

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "hello world"}}]}
        mock_client.post.return_value = mock_resp

        with (
            patch("moment_to_action.models.llm._base.subprocess.Popen"),
            patch("moment_to_action.models.llm._base._wait_for_server"),
            patch("moment_to_action.models.llm._base.httpx.Client", return_value=mock_client),
        ):
            model.load(MagicMock())

        payload = {"messages": [], "max_tokens": 10}
        result = model.run(payload)

        mock_client.post.assert_called_once_with("/v1/chat/completions", json=payload)
        assert result == "hello world"

    def test_run_calls_raise_for_status(self) -> None:
        """run() calls raise_for_status() to propagate HTTP errors."""
        model = _make_model()

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_client.post.return_value = mock_resp

        with (
            patch("moment_to_action.models.llm._base.subprocess.Popen"),
            patch("moment_to_action.models.llm._base._wait_for_server"),
            patch("moment_to_action.models.llm._base.httpx.Client", return_value=mock_client),
        ):
            model.load(MagicMock())

        model.run({"messages": [], "max_tokens": 10})
        mock_resp.raise_for_status.assert_called_once()


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


@pytest.mark.unit
class TestWaitForServer:
    """Tests for the _wait_for_server helper."""

    def test_returns_on_first_200(self) -> None:
        """_wait_for_server returns immediately when server is healthy."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.get.return_value = mock_resp

        with patch("moment_to_action.models.llm._base.time.sleep") as mock_sleep:
            _wait_for_server(mock_client, timeout=5.0)

        mock_client.get.assert_called_once_with("/health")
        mock_sleep.assert_not_called()

    def test_retries_on_connect_error(self) -> None:
        """_wait_for_server retries after ConnectError."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.get.side_effect = [httpx.ConnectError("refused"), mock_resp]

        with patch("moment_to_action.models.llm._base.time.sleep"):
            _wait_for_server(mock_client, timeout=5.0)

        assert mock_client.get.call_count == 2

    def test_raises_on_timeout(self) -> None:
        """_wait_for_server raises RuntimeError when timeout expires."""
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.ConnectError("refused")

        with (
            patch("moment_to_action.models.llm._base.time.sleep"),
            patch(
                "moment_to_action.models.llm._base.time.monotonic",
                side_effect=[0.0, 0.0, 100.0],
            ),
            pytest.raises(RuntimeError, match="healthy"),
        ):
            _wait_for_server(mock_client, timeout=5.0)
