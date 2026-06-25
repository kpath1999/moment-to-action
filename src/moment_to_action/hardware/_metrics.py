"""Pydantic schemas for per-inference metrics returned by inference backends."""

from __future__ import annotations

from pydantic import BaseModel


class LlamaCppInferenceMetrics(BaseModel):
    """Timing metrics returned in the ``timings`` field of a llama.cpp ``/completion`` response.

    Fields map 1:1 to the ``timings`` dict from the native llama.cpp endpoint.

    Attributes:
        prompt_n: Number of tokens in the prompt that were evaluated.
        prompt_ms: Total time spent processing the prompt (milliseconds).
        prompt_per_token_ms: Per-token prompt processing latency (milliseconds).
        prompt_per_second: Prompt processing throughput (tokens/second).
        predicted_n: Number of tokens generated.
        predicted_ms: Total time spent generating tokens (milliseconds).
        predicted_per_token_ms: Per-token generation latency (milliseconds).
        predicted_per_second: Generation throughput (tokens/second).
    """

    prompt_n: int
    prompt_ms: float
    prompt_per_token_ms: float
    prompt_per_second: float
    predicted_n: int
    predicted_ms: float
    predicted_per_token_ms: float
    predicted_per_second: float


InferenceMetrics = LlamaCppInferenceMetrics
"""Union type for all supported inference metric schemas.

Currently only llama.cpp is supported. When additional backends expose
inference metrics, expand this to a proper ``Union``.
"""
