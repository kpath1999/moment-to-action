"""Unit tests for PromptRunner (evaluation + response caching)."""

from __future__ import annotations

import pytest

from moment_to_action.prompt_tuning import (
    EvalDataset,
    KeywordRecallScorer,
    PromptCandidate,
    PromptRunner,
)

from .conftest import ScriptedTarget, make_case


@pytest.mark.unit
class TestPromptRunner:
    """Tests for PromptRunner.evaluate."""

    def test_evaluate_scores_each_case(self) -> None:
        """Evaluate produces one scored result per case."""
        target = ScriptedTarget(responses={"a": "yes", "b": "no"})
        runner = PromptRunner(target=target, scorer=KeywordRecallScorer(), pass_threshold=0.5)
        dataset = EvalDataset(
            [make_case("a", keywords=("yes",)), make_case("b", keywords=("yes",))]
        )

        report = runner.evaluate(PromptCandidate("s", "t"), dataset)

        assert report.num_cases == 2
        assert report.candidate_label == PromptCandidate("s", "t").label
        assert report.scorer_name == "keyword_recall"
        scores = {r.case_id: r.score for r in report.results}
        assert scores == {"a": 1.0, "b": 0.0}

    def test_pass_threshold_applied(self) -> None:
        """A case passes only when its score meets the threshold."""
        target = ScriptedTarget(responses={"a": "yes"})
        runner = PromptRunner(target=target, scorer=KeywordRecallScorer(), pass_threshold=1.0)
        dataset = EvalDataset([make_case("a", keywords=("yes", "definitely"))])

        report = runner.evaluate(PromptCandidate("s", "t"), dataset)

        assert report.results[0].score == 0.5
        assert report.results[0].passed is False

    def test_response_cached_across_identical_content(self) -> None:
        """Two candidates with identical content reuse the cached response."""
        target = ScriptedTarget(responses={"a": "yes"})
        runner = PromptRunner(target=target, scorer=KeywordRecallScorer())
        dataset = EvalDataset([make_case("a", keywords=("yes",))])

        runner.evaluate(PromptCandidate("s", "t", generation=0), dataset)
        runner.evaluate(PromptCandidate("s", "t", generation=1), dataset)

        # Same content hash → generate called once despite two evaluations.
        assert len(target.calls) == 1

    def test_error_recorded_and_not_cached(self) -> None:
        """A failing case records the error, scores 0, and is retried next time."""
        target = ScriptedTarget(responses={"a": "yes"}, fail_case_ids={"a"})
        runner = PromptRunner(target=target, scorer=KeywordRecallScorer())
        dataset = EvalDataset([make_case("a", keywords=("yes",))])
        candidate = PromptCandidate("s", "t")

        report = runner.evaluate(candidate, dataset)
        result = report.results[0]
        assert result.error != ""
        assert result.score == 0.0
        assert result.passed is False
        assert result.response == ""

        # Not cached: a second evaluation attempts the case again.
        runner.evaluate(candidate, dataset)
        assert len(target.calls) == 2

    def test_latency_recorded(self) -> None:
        """Each result carries a non-negative latency measurement."""
        target = ScriptedTarget(responses={"a": "yes"})
        runner = PromptRunner(target=target, scorer=KeywordRecallScorer())
        dataset = EvalDataset([make_case("a", keywords=("yes",))])

        report = runner.evaluate(PromptCandidate("s", "t"), dataset)
        assert report.results[0].latency_ms >= 0.0
