"""Qwen2 1.5B Instruct GGUF model."""

from __future__ import annotations

from moment_to_action.models.llm._base import LlamaGGUFModel


class Qwen2Model(LlamaGGUFModel):
    """Qwen2 1.5B Instruct Q4_0 GGUF model served via llama-server.

    Inherits all subprocess management and HTTP inference logic from
    :class:`~moment_to_action.models.llm._base.LlamaGGUFModel`.  Registered
    in :data:`~moment_to_action.models.MODEL_REGISTRY` under
    ``ModelID.QWEN2_1_5B_INSTRUCT`` so the model manager can download and
    cache the weights automatically.
    """
