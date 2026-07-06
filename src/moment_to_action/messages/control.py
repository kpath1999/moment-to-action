"""Control/sentinel pipeline messages — signal stream structure, carry no payload."""

from __future__ import annotations

from ._base import BaseMessage


class EndOfClipMessage(BaseMessage):
    """Sentinel marking the end of one bounded run of messages.

    Emitted once, after the last message of a bounded run — e.g. the last frame
    of a clip (by whatever produces the frame stream, such as a bench script
    sampling a video file), or the last token of an LLM/VLM generation (by
    :class:`~moment_to_action.stages.llm.LLMStage` /
    :class:`~moment_to_action.stages.vlm.VLMDescriptionStage`). Lets a downstream
    stage flush a running accumulation in real time — one message at a time, with
    no need to buffer the whole run or know its length in advance — without
    polluting every payload message type with an ``end_of_clip``/``done``/``is_last``
    flag.

    Carries no other data — it isn't keyed by clip ID or prompt, since a single
    pipeline processes one bounded run to completion before starting the next, so
    there's never more than one run's sentinel in flight to disambiguate.

    Stages that don't care don't match it in their ``isinstance`` dispatch and it
    falls out of the stream unchanged (or is dropped, depending on the stage's
    ``drop`` predicate). Stages that do care (e.g. an aggregation stage) must
    explicitly pass it through if messages downstream also need to see it.
    """
