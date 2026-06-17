"""Qwen3 4B GGUF model."""

from __future__ import annotations

from moment_to_action.models.llm._base import LlamaGGUFModel


class Qwen3Model(LlamaGGUFModel):
    """Qwen3 4B Q4_K_M GGUF model served via llama-server.

    Inherits all subprocess management and HTTP inference logic from
    :class:`~moment_to_action.models.llm._base.LlamaGGUFModel`.  Registered
    in :data:`~moment_to_action.models.MODEL_REGISTRY` under
    ``ModelID.QWEN3_4B`` so the model manager can download and cache the
    weights automatically.
    """
