"""The target a prompt is evaluated against — i.e. "run the pipeline".

The tuner does not care *how* a prompt turns into a response; it only needs the
:class:`ResponseTarget` protocol.  Today the concrete target wraps a multimodal
GGUF model directly (:class:`VLMResponseTarget`); when a real VLM
:class:`~moment_to_action.stages._base.Stage`/``Pipeline`` lands, a
``PipelineResponseTarget`` can implement the same protocol without touching the
runner, scorer, proposer, or tuner.

The tunable system prompt is folded into the per-request user prompt (see
:meth:`~moment_to_action.prompt_tuning._types.PromptCandidate.compose`).  The
wrapped model is therefore expected to be constructed with an *empty* system
prompt, so swapping candidates never requires reloading model weights or
restarting ``llama-server``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import attrs

if TYPE_CHECKING:
    from moment_to_action.metrics import MetricsCollector

    from ._types import EvalCase, PromptCandidate


@runtime_checkable
class ResponseTarget(Protocol):
    """Produces a text response for a ``(candidate, case)`` pair."""

    def generate(self, candidate: PromptCandidate, case: EvalCase) -> str:
        """Run the candidate prompt on the case and return the response text.

        Args:
            candidate: The prompt candidate to apply.
            case: The evaluation case supplying the question and images.

        Returns:
            The model's raw text response.
        """
        ...


@runtime_checkable
class MultimodalModel(Protocol):
    """Structural type for the subset of a VLM used by :class:`VLMResponseTarget`.

    Satisfied by :class:`~moment_to_action.models.vlm._base.LlamaVLModel` and any
    model exposing the standard ``prepare``/``run``/``post_proc`` inference
    triplet over ``(prompt, images)`` input.
    """

    def prepare(
        self, inputs: tuple[str, list[str]], *, metrics: MetricsCollector | None = ...
    ) -> object:
        """Format ``(prompt, images)`` into a runnable request.

        Args:
            inputs: ``(prompt, b64_images)`` tuple.
            metrics: Optional collector for a preprocess span.

        Returns:
            An opaque prepared-request object for :meth:`run`.
        """
        ...

    def run(self, prepared: object, *, metrics: MetricsCollector | None = ...) -> object:
        """Execute inference on a prepared request.

        Args:
            prepared: The object returned by :meth:`prepare`.
            metrics: Optional collector for an inference span.

        Returns:
            An opaque raw-output object for :meth:`post_proc`.
        """
        ...

    def post_proc(self, raw: object, *, metrics: MetricsCollector | None = ...) -> list[str]:
        """Decode raw output into text responses.

        Args:
            raw: The object returned by :meth:`run`.
            metrics: Optional collector for a post-process span.

        Returns:
            A list of text responses (first element is used).
        """
        ...


@attrs.define
class VLMResponseTarget:
    """A :class:`ResponseTarget` backed by a loaded multimodal GGUF model.

    Attributes:
        model: A loaded multimodal model (empty system prompt — the tuned system
            prompt is composed into the user prompt instead).
        metrics: Optional metrics collector threaded through the model's
            prepare/run/post_proc spans.
    """

    model: MultimodalModel
    metrics: MetricsCollector | None = None

    def generate(self, candidate: PromptCandidate, case: EvalCase) -> str:
        """Compose the candidate prompt and run it through the model.

        Args:
            candidate: The prompt candidate to apply.
            case: The evaluation case supplying the question and images.

        Returns:
            The first decoded text response.

        Raises:
            RuntimeError: If the model returns no responses.
        """
        prompt = candidate.compose(case.question)
        prepared = self.model.prepare((prompt, list(case.images_b64)), metrics=self.metrics)
        raw = self.model.run(prepared, metrics=self.metrics)
        responses = self.model.post_proc(raw, metrics=self.metrics)
        if not responses:
            msg = "model returned no responses"
            raise RuntimeError(msg)
        return responses[0]
