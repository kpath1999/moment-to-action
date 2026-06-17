"""Phi-3.5 Mini Instruct GGUF model."""

from __future__ import annotations

from moment_to_action.models.llm._base import LlamaGGUFModel


class Phi35Model(LlamaGGUFModel):
    """Phi-3.5 Mini Instruct Q4_0 GGUF model served via llama-server.

    Inherits all subprocess management and HTTP inference logic from
    :class:`~moment_to_action.models.llm._base.LlamaGGUFModel`.  Registered
    in :data:`~moment_to_action.models.MODEL_REGISTRY` under
    ``ModelID.PHI35_MINI_INSTRUCT`` so the model manager can download and
    cache the weights automatically.
    """
