"""Evaluate a prompt candidate over a dataset — the "run pipeline → result" step.

:class:`PromptRunner` ties together a
:class:`~moment_to_action.prompt_tuning._target.ResponseTarget` (how to get a
response) and a :class:`~moment_to_action.prompt_tuning._scoring.Scorer` (how
good the response is), producing an
:class:`~moment_to_action.prompt_tuning._types.EvalReport`.

Because VLM inference is slow, responses are cached by
``(content_hash, case_id)``.  Two candidates with identical prompt *content*
therefore reuse cached responses even across generations, so re-proposing an
already-seen prompt is free.  A failed case is not cached, so a transient error
(e.g. server hiccup) can be retried on the next evaluation.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import attrs

from ._types import CaseResult, EvalReport

if TYPE_CHECKING:
    from ._scoring import Scorer
    from ._target import ResponseTarget
    from ._types import EvalCase, EvalDataset, PromptCandidate

logger = logging.getLogger(__name__)


@attrs.define
class PromptRunner:
    """Runs candidates over a dataset and scores the responses.

    Attributes:
        target: Produces a response for each ``(candidate, case)`` pair.
        scorer: Scores each response against its case.
        pass_threshold: Score at or above which a case counts as passed.
    """

    target: ResponseTarget
    scorer: Scorer
    pass_threshold: float = 0.5
    _cache: dict[tuple[str, str], str] = attrs.field(factory=dict, init=False)

    def _response_for(self, candidate: PromptCandidate, case: EvalCase) -> tuple[str, str]:
        """Return ``(response, error)`` for a case, using and filling the cache.

        Successful responses are cached by ``(content_hash, case_id)``; errors
        are returned but not cached so they can be retried later.

        Args:
            candidate: The prompt candidate being evaluated.
            case: The case to generate a response for.

        Returns:
            ``(response, error)`` — ``error`` is ``""`` on success.
        """
        key = (candidate.content_hash, case.case_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached, ""
        try:
            response = self.target.generate(candidate, case)
        except Exception as exc:  # noqa: BLE001 — record per-case failure, keep evaluating
            logger.warning("case %s failed: %s", case.case_id, exc)
            return "", str(exc)
        self._cache[key] = response
        return response, ""

    def evaluate(self, candidate: PromptCandidate, dataset: EvalDataset) -> EvalReport:
        """Evaluate ``candidate`` over every case in ``dataset``.

        Args:
            candidate: The prompt candidate to evaluate.
            dataset: The cases to run it against.

        Returns:
            An :class:`EvalReport` with one :class:`CaseResult` per case.
        """
        results: list[CaseResult] = []
        for case in dataset:
            start = time.perf_counter()
            response, error = self._response_for(candidate, case)
            latency_ms = (time.perf_counter() - start) * 1000.0
            score = 0.0 if error else self.scorer.score(response, case)
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    app=case.app,
                    question=case.question,
                    expected_label=case.expected_label,
                    response=response,
                    score=score,
                    passed=bool(not error and score >= self.pass_threshold),
                    latency_ms=latency_ms,
                    error=error,
                )
            )
        return EvalReport(
            candidate_label=candidate.label,
            scorer_name=self.scorer.name,
            pass_threshold=self.pass_threshold,
            results=tuple(results),
        )
