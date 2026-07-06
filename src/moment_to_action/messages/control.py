"""Control/sentinel pipeline messages — signal stream structure, carry no payload."""

from __future__ import annotations

from ._base import BaseMessage


class EndOfClipMessage(BaseMessage):
    """Sentinel marking the end of one bounded clip's frame stream.

    Emitted once, after the last frame of a clip, by whatever produces the frame
    stream (e.g. a bench script sampling a video file). Lets a downstream stage
    flush a running per-clip accumulation in real time — one frame at a time, with
    no need to buffer the whole clip or know its length in advance — without
    polluting every payload message type with an ``end_of_clip``/``is_last`` flag.

    Stages that don't care about clip boundaries simply don't match it in their
    ``isinstance`` dispatch and it falls out of the stream unchanged (or is
    dropped, depending on the stage's ``drop`` predicate). Stages that do care
    (e.g. an aggregation stage) must explicitly pass it through if messages
    downstream also need to see it.
    """
