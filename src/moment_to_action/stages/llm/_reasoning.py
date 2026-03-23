"""LLM reasoning stage.

ReasoningStage formats YOLO detections into a prompt and runs an LLM.

Input:  DetectionMessage
Output: ReasoningMessage
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.messages import DetectionMessage, ReasoningMessage
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from moment_to_action.messages import Message
    from moment_to_action.models import ModelID, ModelManager

logger = logging.getLogger(__name__)


class ReasoningStage(Stage):
    """Formats YOLO detections into a prompt and runs an LLM.

    Input:  DetectionMessage
    Output: ReasoningMessage
    """

    _backend: ComputeBackend | None
    _handle: object | None

    def __init__(
        self,
        model_id: ModelID | None = None,
        system_prompt: str = "",
        manager: ModelManager | None = None,
    ) -> None:
        super().__init__()
        self._handle = None
        if model_id is not None:
            # Resolve model path through the manager — downloads/caches as needed.
            if manager is None:
                msg = "Model manager is required when a model ID is provided!"
                raise ValueError(msg)

            model_path = manager.get_path(model_id)
            self._backend = ComputeBackend(preferred_unit=ComputeUnit.CPU)
            self._handle = self._backend.load_model(model_path)
            logger.info("ReasoningStage: loaded %s", model_path)
        else:
            self._backend = None
            logger.info("ReasoningStage: running in stub mode (no model loaded)")
        self._system_prompt = system_prompt or (
            "You are analyzing detections from a wearable device. "
            "Based on the detected objects and their positions, assess the scene briefly."
        )

    def _process(self, msg: Message) -> ReasoningMessage | None:
        """Format detections into a prompt and run the LLM."""
        if not isinstance(msg, DetectionMessage):
            err = f"ReasoningStage expects DetectionMessage, got {type(msg).__name__}"
            raise TypeError(err)
        prompt = self._build_prompt(msg)
        # LLM inference — tokenize, run, decode
        # Placeholder until Qwen is wired in
        response = self._run_llm(prompt)
        # latency_ms is stamped by Stage.process() via model_copy
        return ReasoningMessage(
            response=response,
            prompt=prompt,
            timestamp=msg.timestamp,
        )

    def _build_prompt(self, msg: DetectionMessage) -> str:
        lines = [self._system_prompt, "", "Detections:"]
        lines.extend(
            f"  - {box.label} (confidence: {box.confidence:.2f}, "
            f"position: [{box.x1:.0f},{box.y1:.0f},{box.x2:.0f},{box.y2:.0f}])"
            for box in msg.top(5)
        )
        lines.append("\nWhat is happening in this scene?")
        return "\n".join(lines)

    def _run_llm(self, prompt: str) -> str:
        # NOTE(kausar): integrate with Kausar's LLM arch. LLM is a stage that
        # ingests the message, performs inference dispatched via ComputeBackend.
        # For now return the prompt so the pipeline is runnable end-to-end.
        return f"[LLM stub] Received prompt with {len(prompt)} chars."
