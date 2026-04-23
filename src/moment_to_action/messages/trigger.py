"""Trigger messages for accumulated pipeline outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

from ._base import BaseMessage

if TYPE_CHECKING:
    from moment_to_action.messages import Message


class TriggerMessage(BaseMessage):
    """Message emitted by trigger stages with accumulated model outputs."""

    payload: Message
    """The most recent upstream model output."""

    accumulated: list[Message] = Field(default_factory=list)
    """All model outputs collected so far in the pipeline run."""

    fired: bool = True
    """Whether the trigger fired. Always ``True`` for the dummy trigger."""

    trigger_source: str
    """Name of the stage or message type that produced ``payload``."""

