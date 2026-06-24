"""llama.cpp LoadedModel — wraps a llama-server subprocess."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, cast

import httpx

from moment_to_action.hardware._loaded_model import LoadedStreamableModel
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType
from moment_to_action.utils.web import pick_free_port

if TYPE_CHECKING:
    import os
    from collections.abc import Generator

logger = logging.getLogger(__name__)

_HEALTH_TIMEOUT_S = 30.0
_HEALTH_POLL_S = 0.5
_HTTP_OK = 200


def __find_llama_server(explicit: str | os.PathLike[str] | None) -> str:
    """Resolve the path to the ``llama-server`` binary.

    Args:
        explicit: Explicit path provided by the caller. Returned as-is if given.

    Returns:
        Absolute path to the ``llama-server`` binary.

    Raises:
        RuntimeError: If no binary can be located.
    """
    if explicit is not None:
        import os as _os  # noqa: PLC0415

        return _os.fspath(explicit)
    found = shutil.which("llama-server")
    if found is None:
        msg = "llama-server not found. Install it or set llama_server_path in AppConfig."
        raise RuntimeError(msg)
    return found


def _start_llama_model(
    path: str,
    *,
    mmproj: str | None = None,
    server_path: str | os.PathLike[str] | None = None,
    port: int | None = None,
    unit: ComputeUnit,
    cpu_only: bool = False,
) -> LlamaModel:
    """Start a llama-server subprocess and return a :class:`LlamaModel`.

    Args:
        path: Path to the ``.gguf`` model file.
        mmproj: Optional path to the multimodal projector file.
        server_path: Explicit path to the ``llama-server`` binary. If ``None``,
            searches ``PATH``.
        port: Port for llama-server to listen on. If ``None``, a free port is
            assigned automatically.
        unit: Compute unit to report for this model.
        cpu_only: If ``True``, passes ``--ngl 0`` to force CPU-only execution.

    Returns:
        A :class:`LlamaModel` with the server already running and healthy.

    Raises:
        RuntimeError: If ``llama-server`` cannot be found.
        RuntimeError: If the server does not become healthy within 30 seconds.
    """
    resolved_server = __find_llama_server(server_path)
    resolved_port = port if port is not None else pick_free_port()

    args = [
        resolved_server,
        "-m",
        path,
        "--port",
        str(resolved_port),
        "--host",
        "127.0.0.1",
    ]
    if cpu_only:
        args += ["--ngl", "0"]
    if mmproj is not None:
        args += ["--mmproj", mmproj]

    proc = subprocess.Popen(  # noqa: S603
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client = httpx.Client(
        base_url=f"http://127.0.0.1:{resolved_port}",
        timeout=httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0),
    )

    deadline = time.monotonic() + _HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            resp = client.get("/health")
            if resp.status_code == _HTTP_OK:
                break
        except httpx.ConnectError:
            pass
        time.sleep(_HEALTH_POLL_S)
    else:
        proc.terminate()
        proc.wait()
        client.close()
        msg = f"llama-server did not become healthy within {_HEALTH_TIMEOUT_S}s"
        raise RuntimeError(msg)

    model = LlamaModel(
        path=path,
        mmproj=mmproj,
        port=resolved_port,
        server_path=resolved_server,
        unit=unit,
        proc=proc,
        client=client,
    )
    logger.info(
        "LlamaModel: started llama-server (pid=%d, port=%d, model=%s, cpu_only=%s)",
        proc.pid,
        resolved_port,
        path,
        cpu_only,
    )
    return model


class LlamaModel(LoadedStreamableModel):
    """A GGUF model served via llama-server.

    Manages the llama-server subprocess and communicates with it via the
    native llama.cpp ``/completion`` endpoint.

    Use :func:`_start_llama_model` to construct — it handles subprocess
    spawning and health polling.

    Attributes:
        _path: Path to the GGUF model file.
        _mmproj: Path to the multimodal projector file, if any.
        _port: Port the server is listening on.
        _server_path: Path to the ``llama-server`` binary.
        _unit: Compute unit this model is resident on.
        _proc: Running llama-server subprocess.
        _client: httpx client connected to the server.
        _unloaded: Whether :meth:`unload` has been called.
    """

    def __init__(
        self,
        path: str,
        mmproj: str | None,
        port: int,
        server_path: str,
        unit: ComputeUnit,
        proc: subprocess.Popen[bytes],
        client: httpx.Client,
    ) -> None:
        """Initialize a LlamaModel container.

        Args:
            path: Path to the ``.gguf`` model file.
            mmproj: Optional path to the multimodal projector file.
            port: Port the server is listening on.
            server_path: Path to the ``llama-server`` binary.
            unit: Compute unit to report for this model.
            proc: Already-running llama-server subprocess.
            client: httpx client connected to the server.
        """
        self._path = path
        self._mmproj = mmproj
        self._port = port
        self._server_path = server_path
        self._unit = unit
        self._proc: subprocess.Popen[bytes] | None = proc
        self._client: httpx.Client | None = client
        self._unloaded = False

    @property
    def unit(self) -> ComputeUnit:
        """Compute unit this model is resident on."""
        return self._unit

    @property
    def dtype(self) -> DataType:
        """Data type — FP32 (llama-server manages quantization internally)."""
        return DataType.FP32

    @property
    def model_type(self) -> ModelType:
        """Model format — always LLAMA_CPP."""
        return ModelType.LLAMA_CPP

    def run(self, inputs: object) -> object:
        """Run inference via a non-streaming POST to ``/completion``.

        Args:
            inputs: Request body dict with at least a ``"prompt"`` key.

        Returns:
            Generated text as a string.

        Raises:
            RuntimeError: If :meth:`unload` has been called.
            httpx.HTTPStatusError: If the server returns a non-2xx response.
        """
        if self._unloaded or self._client is None:
            msg = "LlamaModel has been unloaded; cannot run inference"
            raise RuntimeError(msg)
        payload = {**cast("dict", inputs), "stream": False}
        resp = self._client.post("/completion", json=payload)
        resp.raise_for_status()
        return str(resp.json()["content"])

    def stream(self, inputs: object) -> Generator[str, None, None]:
        """Stream inference output token by token from ``/completion``.

        Args:
            inputs: Request body dict with at least a ``"prompt"`` key.

        Yields:
            String content chunks as they arrive from the server.

        Raises:
            RuntimeError: If :meth:`unload` has been called.
            httpx.HTTPStatusError: If the server returns a non-2xx response.
        """
        if self._unloaded or self._client is None:
            msg = "LlamaModel has been unloaded; cannot stream inference"
            raise RuntimeError(msg)
        payload = {**cast("dict", inputs), "stream": True}
        with self._client.stream("POST", "/completion", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[len("data: ") :]
                if raw == "[DONE]":
                    break
                chunk = json.loads(raw)
                if chunk.get("content"):
                    yield chunk["content"]
                if chunk.get("stop"):
                    break

    def unload(self) -> None:
        """Terminate llama-server and close the HTTP client.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if not self._unloaded:
            if self._proc is not None:
                self._proc.terminate()
                self._proc.wait()
                self._proc = None
            if self._client is not None:
                self._client.close()
                self._client = None
            self._unloaded = True
