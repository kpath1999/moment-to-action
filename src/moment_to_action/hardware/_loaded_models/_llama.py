"""llama.cpp LoadedModel — wraps a llama-server subprocess."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, cast

import httpx

from moment_to_action.hardware._loaded_model import LoadedStreamableModel
from moment_to_action.hardware._metrics import LlamaCppInferenceMetrics
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType
from moment_to_action.utils.web import pick_free_port

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

_HEALTH_TIMEOUT_S = 120.0
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
    # Check explicit path
    if explicit is not None:
        return os.fspath(explicit)

    # Check $PATH env var
    if found := shutil.which("llama-server"):
        return found

    # Oh no
    msg = "llama-server not found. Install it or set llama_server_path in AppConfig."
    raise RuntimeError(msg)


def _wait_for_health(client: httpx.Client, timeout: float) -> bool:
    """Poll ``/health`` until the server reports ready or the deadline is reached.

    Logs status transitions at INFO level so progress is visible in the logs.

    Args:
        client: httpx client already pointed at the server base URL.
        timeout: Maximum seconds to wait.

    Returns:
        ``True`` if the server became healthy, ``False`` if the deadline elapsed.
    """
    deadline = time.monotonic() + timeout
    t_start = time.monotonic()
    last_status: str = "connecting"

    while time.monotonic() < deadline:
        try:
            resp = client.get("/health")
            if resp.status_code == _HTTP_OK:
                try:
                    body = resp.json()
                    # If body has an explicit "status" key, honour it (llama.cpp returns
                    # "loading model" while weights are being read).  If the body is not
                    # a plain dict or has no "status" key, fall through and accept the 200.
                    status = body.get("status") if isinstance(body, dict) else None
                    if status is None or status == "ok":
                        logger.info(
                            "llama-server healthy (status=%s) after %.1fs",
                            status,
                            time.monotonic() - t_start,
                        )
                        return True
                    if status != last_status:
                        logger.info(
                            "llama-server: status=%s (%.1fs elapsed)",
                            status,
                            time.monotonic() - t_start,
                        )
                        last_status = status
                except Exception:  # noqa: BLE001
                    return True  # unparseable body; assume ready
            elif f"http_{resp.status_code}" != last_status:
                logger.info(
                    "llama-server: HTTP %d (%.1fs elapsed)",
                    resp.status_code,
                    time.monotonic() - t_start,
                )
                last_status = f"http_{resp.status_code}"
        except httpx.ConnectError:
            if last_status != "connecting":
                logger.info(
                    "llama-server: waiting for connection (%.1fs elapsed)",
                    time.monotonic() - t_start,
                )
                last_status = "connecting"

        time.sleep(_HEALTH_POLL_S)

    return False


def _start_llama_model(
    path: str,
    *,
    mmproj: str | None = None,
    server_path: str | os.PathLike[str] | None = None,
    port: int | None = None,
    unit: ComputeUnit,
    cpu_only: bool = False,
    dtype: DataType,
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
        cpu_only: If ``True``, passes ``--ngl 0 --no-mmap`` to force CPU-only execution
            with eager weight loading (avoids mmap lazy-load / spurious empty responses).
        dtype: Data type of this model (e.g. ``DataType.FP32``).

    Returns:
        A :class:`LlamaModel` with the server already running and healthy.

    Raises:
        RuntimeError: If ``llama-server`` cannot be found.
        RuntimeError: If the server does not become healthy within the timeout
            (120 s for GPU, 300 s for CPU).
    """
    # Get server and port
    resolved_server = __find_llama_server(server_path)
    resolved_port = port if port is not None else pick_free_port()

    # Build arguments
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
        # --no-mmap forces eager weight loading so /health only returns 200 after
        # weights are fully resident in RAM, avoiding spurious-empty inference results.
        args += ["--ngl", "0", "--no-mmap"]
    if mmproj is not None:
        args += ["--mmproj", mmproj]

    # Start server
    proc = subprocess.Popen(  # noqa: S603
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Create client for interacting with server
    # Disable read timeout as LLMs have long response times (due to text generation)
    client = httpx.Client(
        base_url=f"http://127.0.0.1:{resolved_port}",
        timeout=httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0),
    )

    # Spin until server is healthy and model is fully loaded.
    if not _wait_for_health(client, _HEALTH_TIMEOUT_S):
        proc.terminate()
        proc.wait()
        client.close()
        msg = f"llama-server did not become healthy within {_HEALTH_TIMEOUT_S}s"
        raise RuntimeError(msg)

    # Build model class
    model = LlamaModel(
        path=path,
        mmproj=mmproj,
        port=resolved_port,
        server_path=resolved_server,
        unit=unit,
        proc=proc,
        client=client,
        dtype=dtype,
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
        _dtype: Data type of this model.
        _proc: Running llama-server subprocess.
        _client: httpx client connected to the server.
        _unloaded: Whether :meth:`unload` has been called.
        _last_inference_metrics: Timing metrics from the most recent inference call.
    """

    def __init__(
        self,
        *,
        path: str,
        mmproj: str | None,
        port: int,
        server_path: str,
        unit: ComputeUnit,
        proc: subprocess.Popen[bytes],
        client: httpx.Client,
        dtype: DataType,
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
            dtype: Data type of this model (e.g. ``DataType.FP32``).
        """
        self._path = path
        self._mmproj = mmproj
        self._port = port
        self._server_path = server_path
        self._unit = unit
        self._dtype = dtype
        self._proc: subprocess.Popen[bytes] | None = proc
        self._client: httpx.Client | None = client
        self._unloaded = False
        self._last_inference_metrics: LlamaCppInferenceMetrics | None = None

    @property
    def unit(self) -> ComputeUnit:
        """Compute unit this model is resident on."""
        return self._unit

    @property
    def dtype(self) -> DataType:
        """Data type of this model."""
        return self._dtype

    @property
    def model_type(self) -> ModelType:
        """Model format — always LLAMA_CPP."""
        return ModelType.LLAMA_CPP

    @property
    def last_inference_metrics(self) -> LlamaCppInferenceMetrics | None:
        """Timing metrics from the most recent :meth:`run` or :meth:`stream` call.

        Returns ``None`` if no inference has been run yet, or if the last
        response did not include a ``timings`` field.

        Returns:
            :class:`LlamaCppInferenceMetrics` from the last inference, or ``None``.
        """
        return self._last_inference_metrics

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
        body = resp.json()
        if timings := body.get("timings"):
            self._last_inference_metrics = LlamaCppInferenceMetrics(**timings)
        return str(body["content"])

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
                    if timings := chunk.get("timings"):
                        self._last_inference_metrics = LlamaCppInferenceMetrics(**timings)
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
