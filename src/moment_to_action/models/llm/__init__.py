"""LLM model classes — language models served via llama-server."""

from __future__ import annotations

from ._base import LlamaGGUFModel
from .qwen2._model import Qwen2Model

__all__ = ["LlamaGGUFModel", "Qwen2Model"]
