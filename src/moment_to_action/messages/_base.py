"""BaseMessage — common base for all pipeline messages."""

from __future__ import annotations

from pydantic import BaseModel


class BaseMessage(BaseModel):
    """Base class for all pipeline messages.

    All pipeline messages carry a timestamp so consumers can
    reason about ordering and latency across pipeline stages.

    Attributes:
        timestamp: Unix epoch timestamp (seconds) when the message was created.
    """

    timestamp: float

    model_config = {"arbitrary_types_allowed": True}  # required for np.ndarray fields
