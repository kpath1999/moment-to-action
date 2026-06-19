"""LlamaServerStage — llama-server subprocess LLM backend.

Starts llama-server as a managed subprocess, downloads the GGUF via
ModelManager, and calls the OpenAI-compatible HTTP API for inference.

Input:  DetectionMessage
Output: ReasoningMessage
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.messages import DetectionMessage, ReasoningMessage
from moment_to_action.metrics._types import SpanType
from moment_to_action.models._model_info import ModelID
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from moment_to_action.config import AppConfig
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector
    from moment_to_action.models import ModelManager
    from moment_to_action.models.llm._base import LlamaGGUFModel

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = "You are a helpful assistant. Be concise."


class LlamaServerStage(Stage):
    """Run an LLM via llama-server for scene reasoning over detections.

    Downloads the GGUF model via :class:`~moment_to_action.models.ModelManager`,
    starts ``llama-server`` as a subprocess in :meth:`__init__`, and sends
    detection descriptions to the OpenAI-compatible ``/v1/chat/completions``
    endpoint.

    Call :meth:`close` (or use as a context manager) to stop the server.

    Args:
        manager: Model manager used to download/cache the GGUF file.
        config: Application configuration supplying ``llama_server_path``
            and ``llama_server_port``.
        model_id: Registry ID of the GGUF model to load.
        system_prompt: System message prepended to every chat request.
        max_tokens: Maximum tokens to generate per completion.

    Raises:
        RuntimeError: If ``config.llama_server_path`` is ``None``.
    """

    _model: LlamaGGUFModel
    _closed: bool

    def __init__(
        self,
        manager: ModelManager,
        config: AppConfig,
        model_id: ModelID = ModelID.QWEN2_1_5B_INSTRUCT,
        system_prompt: str = _DEFAULT_SYSTEM,
        max_tokens: int = 128,
    ) -> None:
        """Initialise and start llama-server.

        Args:
            manager: Model manager used to resolve and download the GGUF.
            config: App config; must have ``llama_server_path`` set.
            model_id: Registry model ID; defaults to Qwen2 1.5B Instruct Q4_0.
            system_prompt: System role message for the chat completion request.
            max_tokens: Maximum tokens the model may generate per call.

        Raises:
            RuntimeError: If ``config.llama_server_path`` is ``None``.
        """
        super().__init__()
        self._closed = True  # guard __del__ before full init
        if config.llama_server_path is None:
            msg = (
                "llama_server_path is not configured. "
                "Set it in AppConfig before creating LlamaServerStage."
            )
            raise RuntimeError(msg)
        self._model = manager.get_model(  # type: ignore[assignment]
            model_id,
            server_path=config.llama_server_path,
            port=config.llama_server_port,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
        self._model.load(ComputeBackend(preferred_unit=ComputeUnit.CPU))
        self._closed = False  # fully initialised — __del__ may now clean up
        logger.info("LlamaServerStage: llama-server started for model %s", model_id.value)

    def _process(self, msg: Message, metrics: MetricsCollector) -> ReasoningMessage | None:
        """Build a prompt from detections, run LLM inference, return a ReasoningMessage.

        Args:
            msg: Input message; must be a :class:`DetectionMessage`.
            metrics: Metrics collector for span timing.

        Returns:
            :class:`ReasoningMessage` with the model's response and prompt.

        Raises:
            TypeError: If ``msg`` is not a :class:`DetectionMessage`.
        """
        if not isinstance(msg, DetectionMessage):
            err = f"LlamaServerStage expects DetectionMessage, got {type(msg).__name__}"
            raise TypeError(err)
        prompt = self._build_prompt(msg)
        with metrics.start_span(SpanType.MODEL_INFERENCE, "llama-server"):
            prepared = self._model.prepare(prompt)
            raw = self._model.run(prepared)
        text = self._model.post_proc(raw)[0]
        return ReasoningMessage(
            response=text,
            prompt=prompt,
            timestamp=msg.timestamp,
        )

    def _build_prompt(self, msg: DetectionMessage) -> str:
        """Format top-5 detections (by confidence) into a user prompt.

        Args:
            msg: Detection message containing object detections.

        Returns:
            Formatted prompt string describing the scene.
        """
        top5 = sorted(msg.detections, key=lambda d: d.confidence, reverse=True)[:5]
        lines = ["Detections:"]
        lines.extend(
            f"  - {d.label} (confidence: {d.confidence:.2f}, "
            f"position: [{d.bbox.x1:.0f},{d.bbox.y1:.0f},{d.bbox.x2:.0f},{d.bbox.y2:.0f}])"
            for d in top5
        )
        lines.append("\nWhat is happening in this scene?")
        return "\n".join(lines)

    def close(self) -> None:
        """Stop llama-server and close the HTTP client.

        Idempotent — safe to call more than once.
        """
        if not self._closed:
            self._model.unload()
            self._closed = True

    def __del__(self) -> None:
        """Best-effort cleanup on garbage collection."""
        self.close()
