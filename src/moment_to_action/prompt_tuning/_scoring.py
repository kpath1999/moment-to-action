"""Scorers that turn a model response into a scalar quality signal.

A scorer maps ``(response, case)`` to a float in ``[0, 1]``.  The runner uses it
to score every case and to decide pass/fail against a threshold.  Scorers are
deliberately simple and dependency-free so they are cheap to run and easy to
reason about; richer judges (e.g. an LLM-as-judge) can be added later by
implementing the :class:`Scorer` protocol.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import attrs

if TYPE_CHECKING:
    from ._types import EvalCase


@runtime_checkable
class Scorer(Protocol):
    """Maps a model response and its case to a score in ``[0, 1]``."""

    @property
    def name(self) -> str:
        """Short identifier recorded in the evaluation report."""
        ...

    def score(self, response: str, case: EvalCase) -> float:
        """Score ``response`` against the expectations encoded in ``case``.

        Args:
            response: The model's raw text output.
            case: The evaluation case, carrying the expected answer.

        Returns:
            A score in ``[0, 1]`` where higher is better.
        """
        ...


@attrs.frozen
class KeywordRecallScorer:
    """Fraction of a case's ``recall_keywords`` present in the response.

    Case-insensitive substring match, mirroring the metric used by
    ``scripts/benchmark_vlms.py``.  A case with no keywords scores ``1.0``.
    """

    @property
    def name(self) -> str:
        """Identifier for this scorer.

        Returns:
            The literal ``"keyword_recall"``.
        """
        return "keyword_recall"

    def score(self, response: str, case: EvalCase) -> float:
        """Compute keyword recall.

        Args:
            response: The model's raw text output.
            case: The evaluation case, whose ``recall_keywords`` are sought.

        Returns:
            Fraction of keywords found in ``[0, 1]``; ``1.0`` when the case has
            no keywords.
        """
        if not case.recall_keywords:
            return 1.0
        lowered = response.lower()
        found = sum(1 for kw in case.recall_keywords if kw.lower() in lowered)
        return found / len(case.recall_keywords)


@attrs.frozen
class LabelMatchScorer:
    """Binary scorer: ``1.0`` if the expected label appears as a whole word.

    Useful for classification tasks whose answers are short tokens
    (``YES`` / ``NO`` / ``COMPLIANT``).  Matching is case-insensitive and
    word-boundary aware so ``"NO"`` does not match inside ``"NOISE"``.  A case
    with an empty ``expected_label`` always scores ``1.0``.
    """

    @property
    def name(self) -> str:
        """Identifier for this scorer.

        Returns:
            The literal ``"label_match"``.
        """
        return "label_match"

    def score(self, response: str, case: EvalCase) -> float:
        """Check whether the expected label is present in the response.

        Args:
            response: The model's raw text output.
            case: The evaluation case, whose ``expected_label`` is sought.

        Returns:
            ``1.0`` if the label is present (or empty), else ``0.0``.
        """
        if not case.expected_label:
            return 1.0
        pattern = rf"\b{re.escape(case.expected_label.lower())}\b"
        return 1.0 if re.search(pattern, response.lower()) else 0.0
