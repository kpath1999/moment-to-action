"""The tuning loop: seed → evaluate → propose → repeat.

:class:`PromptTuner` wires a runner (evaluate a prompt), a proposer (iterate the
prompt), and an optional store (persist the trajectory) into the closed loop
the user described: *prompt → run pipeline → result → iterate prompt*.  The
efficiency win over naive iteration is that the proposer sees the whole scored
:class:`~moment_to_action.prompt_tuning._types.TuningState` (an OPRO-style
optimization trajectory with failure-focused feedback), not just the last
result, so both a human and an LLM hill-climb with memory.

The loop stops when any of these is reached: ``max_iterations`` proposal rounds
completed, ``target_score`` met by the best candidate, or the proposer raises
:class:`~moment_to_action.prompt_tuning._proposers.StopTuning`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import attrs

from ._proposers import StopTuning
from ._types import ScoredCandidate, TuningState

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._proposers import PromptProposer
    from ._runner import PromptRunner
    from ._store import TrajectoryStore
    from ._types import EvalDataset, PromptCandidate

logger = logging.getLogger(__name__)


@attrs.define
class PromptTuner:
    """Drives the evaluate/propose loop over a dataset.

    Attributes:
        runner: Evaluates a candidate over the dataset into a report.
        proposer: Proposes the next candidate from the trajectory.
        dataset: The cases every candidate is evaluated against.
        task_description: Natural-language task/rubric shown to the proposer.
        store: Optional trajectory persistence; ``None`` disables it.
    """

    runner: PromptRunner
    proposer: PromptProposer
    dataset: EvalDataset
    task_description: str
    store: TrajectoryStore | None = None

    def _evaluate_and_record(
        self,
        candidate: PromptCandidate,
        on_report: Callable[[ScoredCandidate], None] | None,
    ) -> ScoredCandidate:
        """Evaluate a candidate, persist it, and fire the report callback.

        Args:
            candidate: The candidate to evaluate.
            on_report: Optional callback invoked with the scored result.

        Returns:
            The scored candidate.
        """
        report = self.runner.evaluate(candidate, self.dataset)
        scored = ScoredCandidate(candidate=candidate, report=report)
        if self.store is not None:
            self.store.record(scored)
        if on_report is not None:
            on_report(scored)
        return scored

    def run(
        self,
        seed: PromptCandidate,
        *,
        max_iterations: int,
        target_score: float | None = None,
        on_report: Callable[[ScoredCandidate], None] | None = None,
    ) -> TuningState:
        """Run the tuning loop starting from ``seed``.

        Args:
            seed: The initial prompt candidate (generation 0).
            max_iterations: Maximum number of proposal rounds after the seed.
            target_score: Stop early once the best mean score reaches this value;
                ``None`` disables early stopping on score.
            on_report: Optional callback invoked with each scored candidate
                (seed and every proposal) as it is produced.

        Returns:
            The final :class:`TuningState` containing the full trajectory.
        """
        scored = self._evaluate_and_record(seed, on_report)
        state = TuningState(task_description=self.task_description, scored=(scored,))

        for iteration in range(max_iterations):
            if target_score is not None and state.best.report.mean_score >= target_score:
                logger.info("target score %.3f reached; stopping", target_score)
                break
            try:
                candidate = self.proposer.propose(state)
            except StopTuning as exc:
                logger.info("proposer stopped the loop: %s", exc or "(no reason given)")
                break
            logger.info("iteration %d: evaluating %s", iteration + 1, candidate.label)
            scored = self._evaluate_and_record(candidate, on_report)
            state = state.append(scored)

        if self.store is not None:
            self.store.write_best(state.best)
        return state
