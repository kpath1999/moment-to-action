"""LLM reasoning stage.

ReasoningStage formats YOLO detections into a prompt and runs an LLM.

Input:  DetectionMessage
Output: ReasoningMessage
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from moment_to_action.hardware import ComputeUnit, LoadedModel, Platform
from moment_to_action.hardware._types import DataType
from moment_to_action.messages import DetectionMessage, ReasoningMessage
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    import os

    from moment_to_action.config import AppConfig
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector
    from moment_to_action.models import ModelID, ModelManager

logger = logging.getLogger(__name__)


def _load_by_extension(platform: Platform, path: os.PathLike[str]) -> LoadedModel:
    """Dispatch to the appropriate platform load method by file extension.

    Args:
        platform: The hardware platform to load on.
        path: Path to the model file.

    Returns:
        A :class:`~moment_to_action.hardware.LoadedModel` for this model.

    Raises:
        ValueError: If the file extension is not recognised.
    """
    p = str(path).lower()
    if p.endswith(".tflite"):
        return platform.load_tflite(ComputeUnit.CPU, path, dtype=DataType.FP32)
    if p.endswith(".onnx"):
        return platform.load_onnx(ComputeUnit.CPU, path, dtype=DataType.FP32)
    if p.endswith(".dlc"):
        return platform.load_dlc(ComputeUnit.CPU, path, dtype=DataType.W8A8)
    msg = f"Unknown model format for ReasoningStage: {path!r}"
    raise ValueError(msg)


class ReasoningStage(Stage):
    """Formats YOLO detections into a prompt and runs an LLM.

    Input:  DetectionMessage
    Output: ReasoningMessage

    Args:
        model_id: Optional model ID to load. If ``None``, runs in stub mode.
        system_prompt: System prompt for the LLM. Defaults to a generic scene description prompt.
        manager: Model manager; required when ``model_id`` is set.
        config: Application configuration; required when ``model_id`` is set.
    """

    _platform: Platform | None
    _handle: LoadedModel | None

    def __init__(
        self,
        model_id: ModelID | None = None,
        system_prompt: str = "",
        manager: ModelManager | None = None,
        config: AppConfig | None = None,
    ) -> None:
        super().__init__()
        self._handle = None
        if model_id is not None:
            # Resolve model path through the manager — downloads/caches as needed.
            if manager is None:
                msg = "Model manager is required when a model ID is provided!"
                raise ValueError(msg)
            if config is None:
                msg = "AppConfig is required when a model ID is provided!"
                raise ValueError(msg)

            model_path = manager.get_path(model_id)
            self._platform = Platform(config)
            self._handle = _load_by_extension(self._platform, model_path)
            logger.info("ReasoningStage: loaded %s", model_path)
        else:
            self._platform = None
            logger.info("ReasoningStage: running in stub mode (no model loaded)")
        self._system_prompt = system_prompt or (
            "You are analyzing detections from a wearable device. "
            "Based on the detected objects and their positions, assess the scene briefly."
        )

    def _process(self, msg: Message, _metrics: MetricsCollector) -> ReasoningMessage | None:
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
        """Format detection results into an LLM prompt.

        Args:
            msg: Detection message containing object detections.

        Returns:
            Formatted prompt string for LLM inference.
        """
        top5 = sorted(msg.detections, key=lambda d: d.confidence, reverse=True)[:5]
        lines = [self._system_prompt, "", "Detections:"]
        lines.extend(
            f"  - {d.label} (confidence: {d.confidence:.2f}, "
            f"position: [{d.bbox.x1:.0f},{d.bbox.y1:.0f},{d.bbox.x2:.0f},{d.bbox.y2:.0f}])"
            for d in top5
        )
        lines.append("\nWhat is happening in this scene?")
        return "\n".join(lines)

    def _run_llm(self, prompt: str) -> str:
        # NOTE(kausar): integrate with Kausar's LLM arch. LLM is a stage that
        # ingests the message, performs inference dispatched via Platform.
        # For now return the prompt so the pipeline is runnable end-to-end.
        return f"[LLM stub] Received prompt with {len(prompt)} chars."
