"""Immutable data types for the prompt-tuning subsystem.

This module holds the pure data types exchanged between the tuning components —
prompt candidates, evaluation cases/datasets, per-case results, aggregate
reports, and the optimization trajectory.  None of these types perform I/O or
model inference; they are the vocabulary the runner, scorer, proposer, and
tuner speak.

The prompt being tuned has two tunable parts:

* ``system_prompt`` — the persona / framing instructions.
* ``task_template`` — a wrapper around the per-case question, optionally
  containing the :data:`QUESTION_PLACEHOLDER` token.

Both are folded into a single user-prompt string by :meth:`PromptCandidate.compose`
so a candidate can be swapped without reloading the underlying model (the model
is constructed with an empty system prompt — see
:class:`~moment_to_action.prompt_tuning._target.VLMResponseTarget`).
"""

from __future__ import annotations

import hashlib
import statistics
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

QUESTION_PLACEHOLDER = "{question}"
"""Token in :attr:`PromptCandidate.task_template` replaced by the case question."""


def _str_tuple(value: Iterable[str]) -> tuple[str, ...]:
    """Coerce an iterable of strings into a tuple (attrs converter).

    Args:
        value: Any iterable of strings.

    Returns:
        The values as a tuple.
    """
    return tuple(value)


def _case_tuple(value: Iterable[EvalCase]) -> tuple[EvalCase, ...]:
    """Coerce an iterable of cases into a tuple (attrs converter).

    Args:
        value: Any iterable of :class:`EvalCase`.

    Returns:
        The cases as a tuple.
    """
    return tuple(value)


def _result_tuple(value: Iterable[CaseResult]) -> tuple[CaseResult, ...]:
    """Coerce an iterable of case results into a tuple (attrs converter).

    Args:
        value: Any iterable of :class:`CaseResult`.

    Returns:
        The results as a tuple.
    """
    return tuple(value)


def _scored_tuple(value: Iterable[ScoredCandidate]) -> tuple[ScoredCandidate, ...]:
    """Coerce an iterable of scored candidates into a tuple (attrs converter).

    Args:
        value: Any iterable of :class:`ScoredCandidate`.

    Returns:
        The scored candidates as a tuple.
    """
    return tuple(value)


@attrs.frozen
class PromptCandidate:
    """A single prompt under evaluation.

    A candidate is defined entirely by its content (``system_prompt`` +
    ``task_template``); the ``generation``, ``parent_id`` and ``rationale``
    fields are lineage metadata that never affect behaviour.  Two candidates
    with identical content share a :attr:`content_hash`, which the runner uses
    as a cache key so re-proposing the same prompt costs nothing.

    Attributes:
        system_prompt: Persona / framing instructions prepended to every prompt.
        task_template: Wrapper around the per-case question.  If it contains
            :data:`QUESTION_PLACEHOLDER`, the token is replaced by the question;
            otherwise the question is appended.
        generation: Iteration number that produced this candidate (0 = seed).
        parent_id: :attr:`label` of the candidate this was derived from, or
            ``""`` for the seed.
        rationale: Free-text explanation of why this candidate was proposed
            (filled in by the proposer; useful for the trajectory log).
    """

    system_prompt: str
    task_template: str
    generation: int = 0
    parent_id: str = ""
    rationale: str = ""

    @property
    def content_hash(self) -> str:
        """Stable 12-char hex digest of the tunable content.

        Returns:
            SHA-256 digest (truncated) of ``system_prompt`` and
            ``task_template``.  Independent of lineage metadata.
        """
        digest = hashlib.sha256(f"{self.system_prompt}\x00{self.task_template}".encode())
        return digest.hexdigest()[:12]

    @property
    def label(self) -> str:
        """Human-readable identifier combining generation and content hash.

        Returns:
            A string like ``"gen03-1a2b3c4d5e6f"``, unique per (generation,
            content) pair.
        """
        return f"gen{self.generation:02d}-{self.content_hash}"

    def render_task(self, question: str) -> str:
        """Render the task template for a specific question.

        Args:
            question: The per-case question to embed.

        Returns:
            ``task_template`` with :data:`QUESTION_PLACEHOLDER` substituted, the
            question appended when no placeholder is present, or the bare
            question when the template is empty.
        """
        if QUESTION_PLACEHOLDER in self.task_template:
            return self.task_template.replace(QUESTION_PLACEHOLDER, question)
        if not self.task_template:
            return question
        return f"{self.task_template}\n\n{question}"

    def compose(self, question: str) -> str:
        """Compose the full user-prompt string for a question.

        The system prompt is folded into the returned string (rather than set
        on the model) so candidates can be swapped without reloading weights.

        Args:
            question: The per-case question to embed.

        Returns:
            The system prompt (if any) followed by the rendered task.
        """
        task = self.render_task(question)
        if self.system_prompt:
            return f"{self.system_prompt}\n\n{task}"
        return task


@attrs.frozen
class EvalCase:
    """One labelled evaluation example.

    Images are carried as opaque base64-encoded JPEG strings so the core stays
    decoupled from rendering/decoding (the driver produces them).

    Attributes:
        case_id: Stable unique identifier used for caching and reporting.
        question: The question posed to the model for this case.
        images_b64: Ordered base64-encoded JPEG frames (no ``data:`` prefix).
        expected_label: The correct answer token (e.g. ``"YES"``).
        recall_keywords: Words a correct answer is expected to contain.
        app: Optional application/group name for per-group reporting.
    """

    case_id: str
    question: str
    images_b64: tuple[str, ...] = attrs.field(converter=_str_tuple)
    expected_label: str = ""
    recall_keywords: tuple[str, ...] = attrs.field(default=(), converter=_str_tuple)
    app: str = ""


@attrs.frozen
class EvalDataset:
    """An ordered collection of evaluation cases.

    Attributes:
        cases: The cases in evaluation order.
    """

    cases: tuple[EvalCase, ...] = attrs.field(converter=_case_tuple)

    def __iter__(self) -> Iterator[EvalCase]:
        """Iterate over the cases.

        Returns:
            An iterator over :class:`EvalCase` instances.
        """
        return iter(self.cases)

    def __len__(self) -> int:
        """Return the number of cases.

        Returns:
            Case count.
        """
        return len(self.cases)

    def filter_by_app(self, app: str) -> EvalDataset:
        """Return a new dataset containing only cases for ``app``.

        Args:
            app: Application/group name to keep.

        Returns:
            A dataset with the matching cases (possibly empty).
        """
        return EvalDataset(tuple(c for c in self.cases if c.app == app))


@attrs.frozen
class CaseResult:
    """The outcome of running one candidate on one case.

    Carries the case's ``question`` and ``expected_label`` so a report is
    self-describing — the proposer can render failures without the dataset.

    Attributes:
        case_id: Identifier of the case this result is for.
        app: Application/group name copied from the case.
        question: The question posed for this case.
        expected_label: The correct answer token for this case.
        response: The model's raw text response (``""`` on error).
        score: Scalar score in ``[0, 1]`` from the active scorer.
        passed: Whether ``score`` met the runner's pass threshold.
        latency_ms: Wall-clock generation latency in milliseconds.
        error: Error message if generation failed, else ``""``.
    """

    case_id: str
    app: str
    question: str
    expected_label: str
    response: str
    score: float
    passed: bool
    latency_ms: float
    error: str = ""


@attrs.frozen
class EvalReport:
    """Aggregate result of evaluating a candidate over a dataset.

    Attributes:
        candidate_label: :attr:`PromptCandidate.label` of the evaluated candidate.
        scorer_name: Name of the scorer used.
        pass_threshold: Score at or above which a case is considered passed.
        results: Per-case results in dataset order.
    """

    candidate_label: str
    scorer_name: str
    pass_threshold: float
    results: tuple[CaseResult, ...] = attrs.field(converter=_result_tuple)

    @property
    def num_cases(self) -> int:
        """Number of cases evaluated.

        Returns:
            Case count.
        """
        return len(self.results)

    @property
    def mean_score(self) -> float:
        """Mean score across all cases.

        Returns:
            The arithmetic mean of case scores, or ``0.0`` when empty.
        """
        if not self.results:
            return 0.0
        return statistics.fmean(r.score for r in self.results)

    @property
    def pass_rate(self) -> float:
        """Fraction of cases that passed the threshold.

        Returns:
            Passed-case fraction in ``[0, 1]``, or ``0.0`` when empty.
        """
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def failures(self) -> tuple[CaseResult, ...]:
        """Failing cases, worst score first.

        Returns:
            Results whose ``passed`` is False, sorted by ascending score.
        """
        failed = [r for r in self.results if not r.passed]
        return tuple(sorted(failed, key=lambda r: r.score))

    def per_app(self) -> dict[str, float]:
        """Mean score grouped by application name.

        Returns:
            Mapping of app name to mean score.  Cases with an empty ``app`` are
            grouped under ``""``.
        """
        groups: dict[str, list[float]] = {}
        for r in self.results:
            groups.setdefault(r.app, []).append(r.score)
        return {app: statistics.fmean(scores) for app, scores in groups.items()}


@attrs.frozen
class ScoredCandidate:
    """A candidate paired with its evaluation report.

    Attributes:
        candidate: The evaluated prompt candidate.
        report: The report produced by evaluating ``candidate``.
    """

    candidate: PromptCandidate
    report: EvalReport


@attrs.frozen
class TuningState:
    """The full optimization trajectory handed to a proposer.

    The proposer (human or LLM) reads this to decide the next candidate; it is
    intentionally free of dataset images so it can be serialized and shown to an
    external model.

    Attributes:
        task_description: Natural-language description of the task and scoring
            rubric, shown to the proposer for context.
        scored: All evaluated candidates in chronological (proposal) order.
    """

    task_description: str
    scored: tuple[ScoredCandidate, ...] = attrs.field(converter=_scored_tuple)

    @property
    def latest(self) -> ScoredCandidate:
        """The most recently evaluated candidate.

        Returns:
            The last :class:`ScoredCandidate` in the trajectory.

        Raises:
            IndexError: If the trajectory is empty.
        """
        return self.scored[-1]

    @property
    def best(self) -> ScoredCandidate:
        """The highest-scoring candidate seen so far.

        Ties are broken in favour of the earliest candidate.

        Returns:
            The :class:`ScoredCandidate` with the greatest mean score.

        Raises:
            ValueError: If the trajectory is empty.
        """
        if not self.scored:
            msg = "TuningState has no scored candidates"
            raise ValueError(msg)
        return max(self.scored, key=lambda sc: sc.report.mean_score)

    def top_k(self, k: int) -> tuple[ScoredCandidate, ...]:
        """The ``k`` highest-scoring candidates, best first.

        Args:
            k: Maximum number of candidates to return.

        Returns:
            Up to ``k`` scored candidates sorted by descending mean score.
        """
        ranked = sorted(self.scored, key=lambda sc: sc.report.mean_score, reverse=True)
        return tuple(ranked[:k])

    def append(self, scored: ScoredCandidate) -> TuningState:
        """Return a new state with ``scored`` appended to the trajectory.

        Args:
            scored: The newly evaluated candidate to record.

        Returns:
            A new :class:`TuningState`; the original is left unchanged.
        """
        return TuningState(self.task_description, (*self.scored, scored))
