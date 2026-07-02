"""Unit tests for the PromptTuner loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from moment_to_action.prompt_tuning import (
    EvalDataset,
    KeywordRecallScorer,
    PromptCandidate,
    PromptRunner,
    PromptTuner,
    ScoredCandidate,
    TrajectoryStore,
)

from .conftest import ScriptedProposer, ScriptedTarget, make_case

if TYPE_CHECKING:
    from pathlib import Path

_TASK = "detect fights"


def _dataset() -> EvalDataset:
    """A two-case dataset keyed on the word 'yes'."""
    return EvalDataset([make_case("a", keywords=("yes",)), make_case("b", keywords=("yes",))])


def _runner(responses: dict[str, str]) -> tuple[PromptRunner, ScriptedTarget]:
    """A runner over a ScriptedTarget returning ``responses``."""
    target = ScriptedTarget(responses=responses)
    return PromptRunner(target=target, scorer=KeywordRecallScorer(), pass_threshold=0.5), target


@pytest.mark.unit
class TestPromptTuner:
    """Tests for PromptTuner.run."""

    def test_seed_only_evaluates_seed_and_persists_best(self, tmp_path: Path) -> None:
        """With max_iterations=0, only the seed is evaluated and persisted."""
        runner, _ = _runner({"a": "yes", "b": "no"})
        proposer = ScriptedProposer(candidates=[])
        reports: list[ScoredCandidate] = []
        tuner = PromptTuner(
            runner=runner,
            proposer=proposer,
            dataset=_dataset(),
            task_description=_TASK,
            store=TrajectoryStore(run_dir=tmp_path),
        )

        state = tuner.run(PromptCandidate("s", "t"), max_iterations=0, on_report=reports.append)

        assert len(state.scored) == 1
        assert len(reports) == 1
        assert (tmp_path / "best.json").exists()
        # Proposer is never consulted when there are no iterations.
        assert proposer.seen_states == []

    def test_proposals_are_evaluated_and_appended(self) -> None:
        """The loop evaluates each proposal and appends it to the trajectory."""
        runner, _ = _runner({"a": "yes", "b": "no"})
        proposer = ScriptedProposer(candidates=[PromptCandidate("better", "t", generation=1)])
        tuner = PromptTuner(
            runner=runner,
            proposer=proposer,
            dataset=_dataset(),
            task_description=_TASK,
        )

        state = tuner.run(PromptCandidate("seed", "t"), max_iterations=5)

        # Seed + one proposal, then StopTuning (proposer exhausted).
        assert len(state.scored) == 2
        assert [sc.candidate.system_prompt for sc in state.scored] == ["seed", "better"]

    def test_target_score_stops_before_first_proposal(self, tmp_path: Path) -> None:
        """A seed already meeting target_score short-circuits before proposing."""
        runner, _ = _runner({"a": "yes", "b": "yes"})
        proposer = ScriptedProposer(candidates=[PromptCandidate("x", "t", generation=1)])
        tuner = PromptTuner(
            runner=runner,
            proposer=proposer,
            dataset=_dataset(),
            task_description=_TASK,
            store=TrajectoryStore(run_dir=tmp_path),
        )

        state = tuner.run(PromptCandidate("seed", "t"), max_iterations=5, target_score=1.0)

        assert len(state.scored) == 1
        assert proposer.seen_states == []

    def test_target_score_stops_after_reaching_it(self) -> None:
        """The loop proposes until target_score is reached, then stops."""
        # Seed responses miss the keyword; proposal responses hit it.
        target = ScriptedTarget()
        target.responses = {"a": "no", "b": "no"}
        runner = PromptRunner(target=target, scorer=KeywordRecallScorer(), pass_threshold=0.5)

        # Flip the target's responses right before the proposal is evaluated.
        class FlipProposer:
            def __init__(self) -> None:
                self.calls = 0

            def propose(self, state: object) -> PromptCandidate:
                del state
                self.calls += 1
                target.responses = {"a": "yes", "b": "yes"}
                return PromptCandidate("improved", "t", generation=1)

        proposer = FlipProposer()
        tuner = PromptTuner(
            runner=runner, proposer=proposer, dataset=_dataset(), task_description=_TASK
        )

        state = tuner.run(PromptCandidate("seed", "t"), max_iterations=5, target_score=1.0)

        # Seed (0.0) + one improved proposal (1.0), then target met → stop.
        assert len(state.scored) == 2
        assert proposer.calls == 1
        assert state.best.report.mean_score == 1.0

    def test_stop_tuning_breaks_loop_without_store(self) -> None:
        """A proposer raising StopTuning ends the loop; store/on_report are optional."""
        runner, _ = _runner({"a": "yes", "b": "no"})
        proposer = ScriptedProposer(candidates=[])  # empty → immediate StopTuning
        tuner = PromptTuner(
            runner=runner,
            proposer=proposer,
            dataset=_dataset(),
            task_description=_TASK,
            store=None,
        )

        state = tuner.run(PromptCandidate("seed", "t"), max_iterations=3)

        assert len(state.scored) == 1
        assert proposer.seen_states  # proposer was consulted once, then stopped
