"""SmolVLM2 GGUF vision-language model."""

from __future__ import annotations

from moment_to_action.models.vlm._base import LlamaVLModel


class SmolVLM2Model(LlamaVLModel):
    """SmolVLM2 GGUF vision-language model served via llama-server.

    Inherits all subprocess management, multimodal prompt building, and HTTP
    inference logic from :class:`~moment_to_action.models.vlm._base.LlamaVLModel`.
    Registered in :data:`~moment_to_action.models.MODEL_REGISTRY` under
    ``ModelID.SMOLVLM2_256M``, ``ModelID.SMOLVLM2_500M``, and
    ``ModelID.SMOLVLM2_2_2B``.
    """
