"""Qwen2.5-VL Instruct GGUF vision-language model."""

from __future__ import annotations

from moment_to_action.models.vlm._base import LlamaVLModel


class Qwen25VLModel(LlamaVLModel):
    """Qwen2.5-VL Instruct GGUF vision-language model served via llama-server.

    Inherits all subprocess management, multimodal prompt building, and HTTP
    inference logic from :class:`~moment_to_action.models.vlm._base.LlamaVLModel`.
    Registered in :data:`~moment_to_action.models.MODEL_REGISTRY` under
    ``ModelID.QWEN25_VL_3B_INSTRUCT`` and ``ModelID.QWEN25_VL_7B_INSTRUCT``.
    """
