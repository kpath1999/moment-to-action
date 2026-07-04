r"""Chat templates and the shared benchmark system prompt.

``CHATML`` and ``PHI3`` are format strings with ``{system}``/``{user}`` placeholders,
applied by :func:`~moment_to_action.prompting._builder.build_payload` for models that
require specific chat tokens. ``None`` (no template) means raw ``system\nuser``
concatenation.
"""

from __future__ import annotations

CHATML = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
"""ChatML template used by Qwen2/Qwen3/Moondream2."""

PHI3 = "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"
"""Phi-3 chat template."""

BENCHMARK_SYSTEM = (
    "You are a scene analysis AI. Answer the user's question directly and concisely. "
    "Lead with your direct answer, then give one sentence of reasoning."
)
"""Shared system prompt used across the LLM/VLM benchmark scenes."""
