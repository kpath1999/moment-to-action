"""Base class for GGUF vision-language models served via llama-server with --mmproj."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import httpx

from moment_to_action.models.llm._base import LlamaGGUFModel, _wait_for_server

if TYPE_CHECKING:
    from pathlib import Path

    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.hardware._types import ComputeUnit
    from moment_to_action.models._formats import ModelFormat


class LlamaVLModel(LlamaGGUFModel):
    """Base for multimodal GGUF vision-language models served via llama-server.

    Extends :class:`~moment_to_action.models.llm._base.LlamaGGUFModel` to:

    - resolve a second ``"mmproj"`` artifact (the vision encoder projection weights)
      from the same variant directory as the text GGUF file.
    - pass ``--mmproj <path>`` to llama-server on startup so the vision tower is loaded.
    - accept ``(prompt, images)`` as input to :meth:`prepare`, where ``images`` is a
      list of base64-encoded JPEG strings, and build the multimodal chat request
      body expected by the OpenAI-compatible ``/v1/chat/completions`` endpoint.

    The three-stage inference pipeline maps to multimodal generation:

    - ``prepare((prompt, b64_images))`` — formats the multimodal chat request body
    - ``run(prepared)`` — sends the HTTP POST, returns the generated text
    - ``post_proc(raw)`` — wraps the text in a list for pipeline compatibility

    Args:
        variant: Registry variant key.
        path: Variant directory containing both the GGUF and mmproj files.
        model_format: File format (``ModelFormat.GGUF``).
        backends: Compute-unit → artifact filename mapping; the first entry
            must contain both a ``"model"`` key (text GGUF) and an ``"mmproj"`` key
            (vision encoder GGUF).
        input_layout: Not applicable to VLMs; expected to be ``None``.
        server_path: Filesystem path to the ``llama-server`` executable.
        port: Port for llama-server to listen on.
        system_prompt: System message prepended to every chat request.
        max_tokens: Maximum tokens the model may generate per call.
        inference_timeout: Read timeout in seconds for ``/v1/chat/completions`` requests.
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
        inference_timeout: float | None = None,
    ) -> None:
        """Initialise with registry metadata, server configuration, and mmproj path.

        Args:
            variant: Registry variant key (e.g. ``"default"``).
            path: Variant directory; both the GGUF and mmproj files are resolved
                relative to this path.
            model_format: File format — should be ``ModelFormat.GGUF``.
            backends: Compute-unit → ``{component_name: filename}`` dict.  Must
                contain at least ``"model"`` and ``"mmproj"`` keys in the first entry.
            input_layout: Unused for VLMs; pass ``None``.
            server_path: Path to the ``llama-server`` executable.
            port: Port for the llama-server HTTP API.
            system_prompt: System message sent in every chat request.
            max_tokens: Maximum tokens to generate per completion.
            inference_timeout: Read timeout in seconds.  ``None`` disables it.
        """
        super().__init__(
            variant,
            path,
            model_format,
            backends=backends,
            input_layout=input_layout,
            server_path=server_path,
            port=port,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            inference_timeout=inference_timeout,
        )
        first_unit_backends = next(iter(backends.values()))
        self._mmproj_path = path / first_unit_backends["mmproj"]

    def load(self, backend: ComputeBackend) -> None:
        """Start llama-server with the vision encoder projection file and wait for health.

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
                "--mmproj",
                str(self._mmproj_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._client = httpx.Client(
            base_url=f"http://127.0.0.1:{self._port}",
            timeout=httpx.Timeout(connect=5.0, read=self._inference_timeout, write=5.0, pool=5.0),
        )
        _wait_for_server(self._client)
        self._backend = backend

    def prepare(self, inputs: tuple[str, list[str]]) -> dict:  # type: ignore[override]
        """Format a prompt and base64-encoded images into a multimodal chat request.

        Args:
            inputs: ``(prompt, b64_images)`` where ``b64_images`` is a list of
                base64-encoded JPEG strings (without the ``data:`` prefix).

        Returns:
            Request body dict suitable for ``/v1/chat/completions``.
        """
        prompt, b64_images = inputs
        image_content: list[dict] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
            for b in b64_images
        ]
        image_content.append({"type": "text", "text": prompt})
        return {
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": image_content},
            ],
            "max_tokens": self._max_tokens,
        }
