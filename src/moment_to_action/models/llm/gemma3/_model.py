"""Gemma 3 GGUF model."""

from __future__ import annotations

from moment_to_action.models.llm._base import LlamaGGUFModel


class Gemma3Model(LlamaGGUFModel):
    """Gemma 3 GGUF model served via llama-server.

    Inherits all subprocess management and HTTP inference logic from
    :class:`~moment_to_action.models.llm._base.LlamaGGUFModel`.  Registered
    in :data:`~moment_to_action.models.MODEL_REGISTRY` under
    ``ModelID.GEMMA3_270M_IT`` and ``ModelID.GEMMA3_1B_IT``.
    """
