"""Unit tests for LlamaVLModel (tested via Qwen25VLModel)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models._formats import ModelFormat
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
_SERVER_PATH = Path("/usr/bin/llama-server")
_SYSTEM = "You are a vision AI."


def _make_model(
    port: int = 8080,
    system_prompt: str = _SYSTEM,
    max_tokens: int = 64,
) -> Qwen25VLModel:
    """Construct a Qwen25VLModel with test parameters."""
    return Qwen25VLModel(
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

    def test_load_starts_subprocess_with_mmproj_arg(self) -> None:
        """load() launches llama-server with --mmproj in the CLI arguments."""
        model = _make_model(port=8080)
        mock_backend = MagicMock()

        with (
            patch("moment_to_action.models.vlm._base.subprocess.Popen") as mock_popen,
            patch("moment_to_action.models.vlm._base._wait_for_server"),
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
                "--mmproj",
                str(_VARIANT_DIR / "mmproj.gguf"),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def test_load_creates_httpx_client(self) -> None:
        """load() creates an httpx.Client pointed at localhost:port."""
        model = _make_model(port=7777)
        mock_backend = MagicMock()

        with (
            patch("moment_to_action.models.vlm._base.subprocess.Popen"),
            patch("moment_to_action.models.vlm._base._wait_for_server"),
            patch("moment_to_action.models.vlm._base.httpx.Client") as mock_client_cls,
        ):
            model.load(mock_backend)

        mock_client_cls.assert_called_once_with(
            base_url="http://127.0.0.1:7777",
            timeout=httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0),
        )

    def test_load_calls_wait_for_server(self) -> None:
        """load() calls _wait_for_server with the httpx client."""
        model = _make_model()
        mock_backend = MagicMock()

        with (
            patch("moment_to_action.models.vlm._base.subprocess.Popen"),
            patch("moment_to_action.models.vlm._base.httpx.Client") as mock_client_cls,
            patch("moment_to_action.models.vlm._base._wait_for_server") as mock_wait,
        ):
            mock_client_instance = MagicMock()
            mock_client_cls.return_value = mock_client_instance
            model.load(mock_backend)

        mock_wait.assert_called_once_with(mock_client_instance)

    def test_load_marks_model_as_loaded(self) -> None:
        """After load(), is_loaded returns True."""
        model = _make_model()

        with (
            patch("moment_to_action.models.vlm._base.subprocess.Popen"),
            patch("moment_to_action.models.vlm._base._wait_for_server"),
        ):
            model.load(MagicMock())

        assert model.is_loaded

    def test_load_raises_if_already_loaded(self) -> None:
        """load() raises RuntimeError when called on an already-loaded model."""
        model = _make_model()

        with (
            patch("moment_to_action.models.vlm._base.subprocess.Popen"),
            patch("moment_to_action.models.vlm._base._wait_for_server"),
        ):
            model.load(MagicMock())
            with pytest.raises(RuntimeError, match="already loaded"):
                model.load(MagicMock())


@pytest.mark.unit
class TestLlamaVLModelPrepare:
    """Tests for LlamaVLModel.prepare()."""

    def test_prepare_returns_dict(self) -> None:
        """prepare() returns a dict."""
        model = _make_model()
        result = model.prepare(("describe this", ["abc123"]))
        assert isinstance(result, dict)

    def test_prepare_system_message(self) -> None:
        """prepare() includes the system prompt as the first message."""
        model = _make_model(system_prompt="You are helpful.")
        result = model.prepare(("question", ["img1"]))
        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}

    def test_prepare_user_message_has_image_url_entries(self) -> None:
        """prepare() includes image_url content blocks for each base64 image."""
        model = _make_model()
        result = model.prepare(("what is this?", ["b64a", "b64b"]))
        user_content = result["messages"][1]["content"]
        image_blocks = [c for c in user_content if c.get("type") == "image_url"]
        assert len(image_blocks) == 2
        assert image_blocks[0]["image_url"]["url"] == "data:image/jpeg;base64,b64a"
        assert image_blocks[1]["image_url"]["url"] == "data:image/jpeg;base64,b64b"

    def test_prepare_user_message_has_text_entry(self) -> None:
        """prepare() includes a text content block with the prompt."""
        model = _make_model()
        result = model.prepare(("what is this?", ["b64a"]))
        user_content = result["messages"][1]["content"]
        text_blocks = [c for c in user_content if c.get("type") == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "what is this?"

    def test_prepare_max_tokens(self) -> None:
        """prepare() includes max_tokens from constructor."""
        model = _make_model(max_tokens=999)
        result = model.prepare(("x", []))
        assert result["max_tokens"] == 999

    def test_prepare_no_images(self) -> None:
        """prepare() handles an empty image list (text-only fallback)."""
        model = _make_model()
        result = model.prepare(("text only", []))
        user_content = result["messages"][1]["content"]
        image_blocks = [c for c in user_content if c.get("type") == "image_url"]
        text_blocks = [c for c in user_content if c.get("type") == "text"]
        assert len(image_blocks) == 0
        assert len(text_blocks) == 1
