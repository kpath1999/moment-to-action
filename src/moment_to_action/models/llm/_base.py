"""Base class for GGUF language models served via llama-server."""

from __future__ import annotations

import logging
import subprocess
import time
from typing import TYPE_CHECKING

import httpx

from moment_to_action.models._base import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.hardware._types import ComputeUnit
    from moment_to_action.models._formats import ModelFormat

logger = logging.getLogger(__name__)

_HEALTH_TIMEOUT_S = 30.0
_HEALTH_POLL_S = 0.5
_HTTP_OK = 200


def _wait_for_server(client: httpx.Client, timeout: float = _HEALTH_TIMEOUT_S) -> None:
    """Poll GET /health until the server responds 200 or timeout expires.

    Args:
        client: httpx client pointed at the llama-server base URL.
        timeout: Maximum seconds to wait before raising.

    Raises:
        RuntimeError: If the server does not become healthy within ``timeout``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = client.get("/health")
            if resp.status_code == _HTTP_OK:
                return
        except httpx.ConnectError:
            pass
        time.sleep(_HEALTH_POLL_S)
    msg = f"llama-server did not become healthy within {timeout}s"
    raise RuntimeError(msg)


class LlamaGGUFModel(BaseModel[str, dict, str, str]):
    """Base for GGUF language models served via llama-server.

    Manages the llama-server subprocess and an httpx client for calling
    the OpenAI-compatible ``/v1/chat/completions`` endpoint.

    The three-stage inference pipeline maps to LLM text generation:

    - ``prepare(prompt)`` — formats the chat request body (messages + params)
    - ``run(prepared)`` — sends the HTTP POST, returns the generated text
    - ``post_proc(raw)`` — wraps the text in a list for pipeline compatibility

    Args:
        variant: Registry variant key.
        path: Variant directory containing the GGUF file.
        model_format: File format (``ModelFormat.GGUF``).
        backends: Compute-unit → artifact filename mapping; the first entry
            must contain a ``"model"`` key naming the ``.gguf`` file.
            Annotate as ``GPU`` since llama-server targets GPU internally.
        input_layout: Not applicable to LLMs; expected to be ``None``.
        server_path: Filesystem path to the ``llama-server`` executable.
        port: Port for llama-server to listen on (and for the client to connect).
        system_prompt: System message prepended to every chat request.
        max_tokens: Maximum tokens the model may generate per call.
    """

    def __init__(
        self,
        variant: str,
        path: Path,
        model_format: ModelFormat | None = None,
        *,
        backends: dict[ComputeUnit, dict[str, str]],
        input_layout: str | None = None,
        server_path: Path,
        port: int = 8080,
        system_prompt: str = "",
        max_tokens: int = 128,
    ) -> None:
        """Initialise with registry metadata and server configuration.

        Args:
            variant: Registry variant key (e.g. ``"default"``).
            path: Variant directory; the GGUF file is at
                ``path / next(iter(backends.values()))["model"]``.
            model_format: File format — should be ``ModelFormat.GGUF``.
            backends: Compute-unit → ``{component_name: filename}`` dict.
            input_layout: Unused for LLMs; pass ``None``.
            server_path: Path to the ``llama-server`` executable.
            port: Port for the llama-server HTTP API.
            system_prompt: System message sent in every chat request.
            max_tokens: Maximum tokens to generate per completion.
        """
        super().__init__(
            variant,
            path,
            model_format,
            backends=backends,
            input_layout=input_layout,
        )
        # llama-server manages its own GPU/CPU dispatch internally; the registry
        # annotates the compute unit that the server actually targets.  We resolve
        # the GGUF filename from whichever unit is listed first rather than
        # hard-coding CPU so GPU-annotated entries work correctly.
        self._gguf_path = path / next(iter(backends.values()))["model"]
        self._server_path = server_path
        self._port = port
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._server_proc: subprocess.Popen[bytes] | None = None
        self._client: httpx.Client | None = None

    def load(self, backend: ComputeBackend) -> None:
        """Start llama-server and wait for it to become healthy.

        Args:
            backend: Unused — llama-server runs independently of ComputeBackend.

        Raises:
            RuntimeError: If the model is already loaded.
            RuntimeError: If llama-server does not start within the health timeout.
        """
        if self.is_loaded:
            msg = f"{type(self).__name__} is already loaded"
            raise RuntimeError(msg)
        self._server_proc = subprocess.Popen(  # noqa: S603
            [
                str(self._server_path),
                "-m",
                str(self._gguf_path),
                "--port",
                str(self._port),
                "--host",
                "127.0.0.1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._client = httpx.Client(base_url=f"http://127.0.0.1:{self._port}")
        _wait_for_server(self._client)
        self._backend = backend
        logger.info(
            "%s: llama-server started (pid=%d, port=%d, model=%s)",
            type(self).__name__,
            self._server_proc.pid,
            self._port,
            self._gguf_path.name,
        )

    def unload(self) -> None:
        """Terminate llama-server and close the HTTP client.

        Safe to call when not loaded (no-op).
        """
        if self._server_proc is not None:
            self._server_proc.terminate()
            self._server_proc.wait()
            self._server_proc = None
        if self._client is not None:
            self._client.close()
            self._client = None
        self._backend = None

    def prepare(self, inputs: str) -> dict:
        """Format a user prompt into a chat completion request body.

        Args:
            inputs: User-facing text prompt.

        Returns:
            Request body dict suitable for ``/v1/chat/completions``.
        """
        return {
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": inputs},
            ],
            "max_tokens": self._max_tokens,
        }

    def run(self, prepared: dict) -> str:
        """Send the chat request and return the generated text.

        Args:
            prepared: Request body from :meth:`prepare`.

        Returns:
            Generated text content from ``choices[0].message.content``.

        Raises:
            RuntimeError: If the model has not been loaded.
            httpx.HTTPStatusError: If the server returns a non-2xx response.
        """
        if self._client is None:
            msg = f"{type(self).__name__} is not loaded; call load() first"
            raise RuntimeError(msg)
        resp = self._client.post("/v1/chat/completions", json=prepared)
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"])

    def post_proc(self, raw: str) -> list[str]:
        """Wrap the generated text in a list for pipeline compatibility.

        Args:
            raw: Text returned by :meth:`run`.

        Returns:
            Single-element list containing the generated text.
        """
        return [raw]

    def verify_outputs(
        self,
        inputs: object,
        ref_outputs: object,
        *,
        tol: float,
        is_npu: bool,
    ) -> tuple[bool, str]:
        """Not supported — llama-server does not expose per-tensor verification.

        Raises:
            NotImplementedError: Always.
        """
        msg = (
            f"{type(self).__name__} does not support verify_outputs; "
            "llama-server handles inference internally."
        )
        raise NotImplementedError(msg)
