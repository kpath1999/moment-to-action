"""BaseMessage — common base for all pipeline messages."""

from __future__ import annotations

from pydantic import BaseModel


class BaseMessage(BaseModel):
    """Base class for all pipeline messages."""

    model_config = {"arbitrary_types_allowed": True}  # required for NDArray fields

    timestamp: float
    """Unix epoch timestamp (seconds) when the message was created."""

    latency_ms: float = 0.0
    """End-to-end latency in milliseconds from capture to this message."""
