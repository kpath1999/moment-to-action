"""DecisionStage — interprets a grammar-constrained GenerationMessage stream as yes/no."""

from __future__ import annotations

from typing import TYPE_CHECKING

from moment_to_action.benchmarking import detect_yn
from moment_to_action.messages.llm import (
    DecisionMessage,
    DecisionReasoningMessage,
    GenerationMessage,
)
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moment_to_action.messages import Message
    from moment_to_action.metrics import MetricsCollector

_DECISION_PREFIXES = ("YES", "NO")


def _strip_decision_prefix(text: str) -> str:
    """Strip a leading ``YES``/``NO`` token (and following punctuation) from *text*.

    Args:
        text: Accumulated response text, expected to start with the
            grammar-forced ``YES``/``NO`` literal.

    Returns:
        *text* with the leading decision token and any following punctuation/
        whitespace removed.
    """
    stripped = text
    for word in _DECISION_PREFIXES:
        if stripped.startswith(word):
            stripped = stripped[len(word) :]
            break
    return stripped.lstrip(" ,.:;!?\n")


class DecisionStage(Stage):
    """Interprets a streamed, grammar-constrained LLM generation as a yes/no decision.

    Consumes the :class:`~moment_to_action.messages.llm.GenerationMessage` stream
    produced by an upstream ``LLMStage(model, grammar=YES_NO_GRAMMAR, ...)`` (see
    :data:`~moment_to_action.prompting.YES_NO_GRAMMAR`). As soon as the
    grammar-forced leading ``YES``/``NO`` is unambiguous in the accumulated
    response text, emits a :class:`~moment_to_action.messages.llm.DecisionMessage`
    immediately, then forwards the remaining response text as
    :class:`~moment_to_action.messages.llm.DecisionReasoningMessage` partials.

    A sink that only needs the verdict can stop pulling right after the
    ``DecisionMessage`` — the resulting ``GeneratorExit`` aborts the rest of
    generation upstream, at the cost of only the tokens generated so far.
    """

    def __init__(self, *, metrics: MetricsCollector | None = None) -> None:
        """Initialize the stage with an empty per-prompt decision-tracking state.

        Args:
            metrics: Metrics collector used to time this stage's execution.
        """
        super().__init__(window=1, metrics=metrics)
        self._decided_prompts: set[str] = set()

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Interpret one buffered ``GenerationMessage`` partial.

        Args:
            items: Single-element window containing the incoming
                :class:`~moment_to_action.messages.llm.GenerationMessage`.

        Yields:
            A :class:`~moment_to_action.messages.llm.DecisionMessage` the first
            time a decision becomes unambiguous for this prompt, followed by
            zero or more :class:`~moment_to_action.messages.llm.DecisionReasoningMessage`
            partials on this and subsequent calls for the same prompt.
        """
        msg = items[0]
        if not isinstance(msg, GenerationMessage) or msg.type != "response":
            return

        already_decided = msg.prompt in self._decided_prompts
        if not already_decided:
            decision = detect_yn(msg.text)
            if decision is None:
                return
            self._decided_prompts.add(msg.prompt)
            yield DecisionMessage(timestamp=msg.timestamp, decision=decision, prompt=msg.prompt)

        yield DecisionReasoningMessage(
            timestamp=msg.timestamp,
            text=_strip_decision_prefix(msg.text),
            prompt=msg.prompt,
            done=msg.done,
        )

        if msg.done:
            self._decided_prompts.discard(msg.prompt)
