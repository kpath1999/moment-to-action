"""LLMStage — streams LLM generation over a detection-derived prompt."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from moment_to_action.benchmarking import detect_yn
from moment_to_action.messages.detection import DetectionMessage
from moment_to_action.messages.llm import EndOfGenerationMessage, GenerationMessage
from moment_to_action.prompting import build_detection_prompt
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moment_to_action.hardware import ComputeUnit, Platform
    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector
    from moment_to_action.models.llm._base import LlamaGGUFModel

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

_Phase = Literal["think", "response"]


def _partial_tag_hold_len(buf: str, tag: str) -> int:
    """Return how many trailing characters of *buf* could be the start of *tag*.

    Used to avoid emitting a tag that has been split across two token chunks:
    the held-back suffix is re-checked once the next chunk arrives.

    Args:
        buf: Accumulated raw text not yet classified.
        tag: The opening or closing tag to check against.

    Returns:
        Length of the longest suffix of *buf* that is a proper prefix of *tag*
        (0 if none, and never more than ``len(tag) - 1``).
    """
    max_check = min(len(buf), len(tag) - 1)
    for length in range(max_check, 0, -1):
        if tag.startswith(buf[-length:]):
            return length
    return 0


class _ThinkResponseRouter:
    """Splits a raw token stream into "think"/"response" phases at ``<think>`` tags.

    Starts in the "response" phase. Models that never emit a ``<think>`` block
    (including all VLMs) stay in "response" for their entire output. Models that
    do emit one switch to "think" when ``<think>`` arrives and back to "response"
    at ``</think>``; the tags themselves are swallowed, never appearing in the
    routed text.

    Handles a tag split across two token chunks (or multiple phase switches
    within one chunk) by holding back a suffix that could be a partial tag
    prefix until the next chunk resolves it.
    """

    def __init__(self) -> None:
        """Initialize in the "response" phase with an empty pending buffer."""
        self._phase: _Phase = "response"
        self._buf = ""

    @property
    def phase(self) -> _Phase:
        """The current phase, updated as tags are consumed."""
        return self._phase

    def feed(self, token: str) -> list[tuple[_Phase, str]]:
        """Feed one raw token chunk, returning routed ``(phase, text)`` segments.

        A single token may contain zero, one, or multiple phase transitions;
        each returned segment is attributed to the phase it actually belongs to.

        Args:
            token: Newly arrived raw text chunk from the model stream.

        Returns:
            List of ``(phase, text)`` segments to append to that phase's
            accumulated buffer, in order. Empty if the token was entirely a
            tag or a held-back partial tag.
        """
        self._buf += token
        segments: list[tuple[_Phase, str]] = []
        while True:
            tag = _THINK_CLOSE if self._phase == "think" else _THINK_OPEN
            idx = self._buf.find(tag)
            if idx == -1:
                hold = _partial_tag_hold_len(self._buf, tag)
                split = len(self._buf) - hold
                segment, self._buf = self._buf[:split], self._buf[split:]
                if segment:
                    segments.append((self._phase, segment))
                return segments
            segment, self._buf = self._buf[:idx], self._buf[idx + len(tag) :]
            if segment:
                segments.append((self._phase, segment))
            self._phase = "response" if self._phase == "think" else "think"


class LLMStage(Stage):
    """Streams a language model's response to a detection-derived prompt.

    Consumes a :class:`~moment_to_action.messages.detection.DetectionMessage` and
    streams the model's generation token by token as
    :class:`~moment_to_action.messages.llm.GenerationMessage` partials, splitting
    ``<think>...</think>`` reasoning (if the model emits one) from the final
    response text. A downstream sink that stops pulling (e.g. once a decision
    fires — see :class:`~moment_to_action.stages.llm.DecisionStage`) closes the
    underlying model stream via ``GeneratorExit``, aborting the rest of generation.

    *system_prompt* and *max_tokens* are configured on *model* at construction
    (``ModelManager.get_model(..., system_prompt=..., max_tokens=...)``); the
    per-message *question* comes from
    :attr:`~moment_to_action.messages.detection.DetectionMessage.question`, so one
    loaded model/stage instance can serve any question — it isn't fixed at
    construction.
    """

    def __init__(
        self,
        model: LlamaGGUFModel,
        *,
        grammar: str | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialize the stage with a language model.

        Args:
            model: A loaded :class:`~moment_to_action.models.llm._base.LlamaGGUFModel`.
            grammar: Optional GBNF grammar constraining generation (e.g.
                :data:`~moment_to_action.prompting.YES_NO_GRAMMAR`).
            metrics: Metrics collector used to time this stage and record
                per-token ttft/itl via ``MetricsCollector.timed_stream``.
        """
        super().__init__(window=1, metrics=metrics)
        self._model = model
        self._grammar = grammar

    def load(self, platform: Platform, unit: ComputeUnit | None = None) -> None:
        """Load the wrapped model onto *platform*.

        Args:
            platform: The hardware platform to load onto.
            unit: The compute unit to target.

        Raises:
            ValueError: If *unit* is ``None``.
        """
        if unit is None:
            msg = "LLMStage.load requires a compute unit"
            raise ValueError(msg)
        self._model.load(platform, unit)

    def unload(self) -> None:
        """Unload the wrapped model."""
        self._model.unload()

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Stream the model's response to the detection-derived prompt.

        Args:
            items: Single-element window containing the incoming
                :class:`~moment_to_action.messages.detection.DetectionMessage`.

        Yields:
            Partial :class:`~moment_to_action.messages.llm.GenerationMessage`
            objects, one per token, a final one with the complete response text,
            then an :class:`~moment_to_action.messages.llm.EndOfGenerationMessage`.
        """
        msg = items[0]
        if not isinstance(msg, DetectionMessage):
            return

        prompt = build_detection_prompt(msg.detections, msg.question)
        router = _ThinkResponseRouter()
        think, resp = "", ""

        for token in self._metrics.timed_stream(
            self._model.stream(prompt, grammar=self._grammar), yn_predicate=detect_yn
        ):
            for phase, segment in router.feed(token):
                if phase == "think":
                    think += segment
                else:
                    resp += segment
            phase = router.phase
            yield GenerationMessage(
                timestamp=msg.timestamp,
                prompt=prompt,
                text=think if phase == "think" else resp,
                type=phase,
            )

        yield GenerationMessage(
            timestamp=msg.timestamp,
            prompt=prompt,
            text=resp,
            type="response",
        )
        yield EndOfGenerationMessage(timestamp=msg.timestamp, prompt=prompt)
