"""Qwen3-VL Instruct GGUF vision-language model."""

from __future__ import annotations

from moment_to_action.models.vlm._base import LlamaVLModel


class Qwen3VLModel(LlamaVLModel):
    """Qwen3-VL Instruct GGUF vision-language model served via llama-server.

    Inherits all subprocess management, multimodal prompt building, and HTTP
    inference logic from :class:`~moment_to_action.models.vlm._base.LlamaVLModel`.
    Registered in :data:`~moment_to_action.models.MODEL_REGISTRY` under
    ``ModelID.QWEN3_VL_2B_INSTRUCT`` and ``ModelID.QWEN3_VL_4B_INSTRUCT``.
    """
