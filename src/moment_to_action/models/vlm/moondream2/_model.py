"""Moondream2 GGUF vision-language model."""

from __future__ import annotations

from moment_to_action.models.vlm._base import LlamaVLModel


class Moondream2Model(LlamaVLModel):
    """Moondream2 GGUF vision-language model served via llama-server.

    Inherits all subprocess management, multimodal prompt building, and HTTP
    inference logic from :class:`~moment_to_action.models.vlm._base.LlamaVLModel`.
    Registered in :data:`~moment_to_action.models.MODEL_REGISTRY` under
    ``ModelID.MOONDREAM2``.
    """
