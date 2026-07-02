"""Unit tests for prompt_tuning data types."""

from __future__ import annotations

import pytest

from moment_to_action.prompt_tuning import (
    QUESTION_PLACEHOLDER,
    CaseResult,
    EvalCase,
    EvalDataset,
    EvalReport,
    PromptCandidate,
)

from .conftest import make_case, make_report, make_result, make_scored, make_state


def make_report_from(results: list[CaseResult]) -> EvalReport:
    """Build a report attached to a throwaway candidate for report-only tests."""
    return make_report(PromptCandidate("sys", "tmpl"), results)


@pytest.mark.unit
class TestPromptCandidate:
    """Tests for PromptCandidate content hashing, labels, and rendering."""

    def test_content_hash_ignores_lineage_metadata(self) -> None:
        """content_hash depends only on system_prompt + task_template."""
        a = PromptCandidate("sys", "tmpl", generation=0, parent_id="", rationale="x")
        b = PromptCandidate("sys", "tmpl", generation=9, parent_id="p", rationale="y")
        assert a.content_hash == b.content_hash

    def test_content_hash_changes_with_content(self) -> None:
        """Different content yields a different hash."""
        a = PromptCandidate("sys", "tmpl")
        b = PromptCandidate("sys", "other")
        assert a.content_hash != b.content_hash

    def test_label_combines_generation_and_hash(self) -> None:
        """Label is gen<NN>-<hash>."""
        c = PromptCandidate("sys", "tmpl", generation=3)
        assert c.label == f"gen03-{c.content_hash}"

    def test_render_task_substitutes_placeholder(self) -> None:
        """A template with the placeholder has it replaced by the question."""
        c = PromptCandidate("", f"Prefix. {QUESTION_PLACEHOLDER} Suffix.")
        assert c.render_task("Q?") == "Prefix. Q? Suffix."

    def test_render_task_appends_when_no_placeholder(self) -> None:
        """A non-empty template without the placeholder gets the question appended."""
        c = PromptCandidate("", "Framing.")
        assert c.render_task("Q?") == "Framing.\n\nQ?"

    def test_render_task_returns_question_when_template_empty(self) -> None:
        """An empty template renders the bare question."""
        c = PromptCandidate("", "")
        assert c.render_task("Q?") == "Q?"

    def test_compose_prepends_system_prompt(self) -> None:
        """Compose prefixes the system prompt when present."""
        c = PromptCandidate("You are X.", QUESTION_PLACEHOLDER)
        assert c.compose("Q?") == "You are X.\n\nQ?"

    def test_compose_omits_empty_system_prompt(self) -> None:
        """Compose returns just the task when the system prompt is empty."""
        c = PromptCandidate("", QUESTION_PLACEHOLDER)
        assert c.compose("Q?") == "Q?"


@pytest.mark.unit
class TestEvalCaseAndDataset:
    """Tests for EvalCase converters and EvalDataset helpers."""

    def test_case_coerces_iterables_to_tuples(self) -> None:
        """List inputs for images/keywords are stored as tuples."""
        case = EvalCase("c", "Q?", ["a", "b"], "YES", ["yes"])
        assert case.images_b64 == ("a", "b")
        assert case.recall_keywords == ("yes",)

    def test_dataset_len_and_iter(self) -> None:
        """EvalDataset supports len() and iteration in order."""
        cases = [make_case("a"), make_case("b")]
        ds = EvalDataset(cases)
        assert len(ds) == 2
        assert [c.case_id for c in ds] == ["a", "b"]

    def test_dataset_filter_by_app(self) -> None:
        """filter_by_app keeps only matching cases."""
        ds = EvalDataset([make_case("a", app="x"), make_case("b", app="y")])
        filtered = ds.filter_by_app("y")
        assert [c.case_id for c in filtered] == ["b"]


@pytest.mark.unit
class TestEvalReport:
    """Tests for EvalReport aggregate metrics."""

    def test_empty_report_metrics_are_zero(self) -> None:
        """An empty report reports zero mean and pass rate."""
        report = make_report_from([])
        assert report.num_cases == 0
        assert report.mean_score == 0.0
        assert report.pass_rate == 0.0

    def test_mean_score_and_pass_rate(self) -> None:
        """mean_score averages scores; pass_rate counts passed cases."""
        report = make_report_from(
            [
                make_result("a", score=1.0, passed=True),
                make_result("b", score=0.0, passed=False),
            ]
        )
        assert report.mean_score == 0.5
        assert report.pass_rate == 0.5

    def test_failures_sorted_worst_first(self) -> None:
        """Failures contains only failed cases, ascending by score."""
        report = make_report_from(
            [
                make_result("pass", score=1.0, passed=True),
                make_result("mid", score=0.4, passed=False),
                make_result("worst", score=0.1, passed=False),
            ]
        )
        assert [r.case_id for r in report.failures] == ["worst", "mid"]

    def test_per_app_groups_scores(self) -> None:
        """per_app returns the mean score per application."""
        report = make_report_from(
            [
                make_result("a", score=1.0, app="x"),
                make_result("b", score=0.0, app="x"),
                make_result("c", score=0.5, app="y"),
            ]
        )
        assert report.per_app() == {"x": 0.5, "y": 0.5}


@pytest.mark.unit
class TestTuningState:
    """Tests for TuningState trajectory helpers."""

    def test_latest_returns_last_appended(self) -> None:
        """Latest returns the most recently added scored candidate."""
        s1 = make_scored(PromptCandidate("a", "t"), [make_result(score=0.2)])
        s2 = make_scored(PromptCandidate("b", "t"), [make_result(score=0.9)])
        state = make_state(s1, s2)
        assert state.latest is s2

    def test_best_returns_highest_mean_score(self) -> None:
        """Best returns the candidate with the greatest mean score."""
        low = make_scored(PromptCandidate("a", "t"), [make_result(score=0.2)])
        high = make_scored(PromptCandidate("b", "t"), [make_result(score=0.9)])
        state = make_state(low, high)
        assert state.best is high

    def test_best_breaks_ties_toward_earliest(self) -> None:
        """On equal scores, best keeps the earliest candidate."""
        first = make_scored(PromptCandidate("a", "t"), [make_result(score=0.5)])
        second = make_scored(PromptCandidate("b", "t"), [make_result(score=0.5)])
        state = make_state(first, second)
        assert state.best is first

    def test_best_raises_on_empty_state(self) -> None:
        """Best raises ValueError when there is no trajectory."""
        state = make_state()
        with pytest.raises(ValueError, match="no scored candidates"):
            _ = state.best

    def test_top_k_returns_best_first(self) -> None:
        """top_k returns up to k candidates sorted by descending mean score."""
        a = make_scored(PromptCandidate("a", "t"), [make_result(score=0.2)])
        b = make_scored(PromptCandidate("b", "t"), [make_result(score=0.9)])
        c = make_scored(PromptCandidate("c", "t"), [make_result(score=0.5)])
        state = make_state(a, b, c)
        top = state.top_k(2)
        assert [sc.report.mean_score for sc in top] == [0.9, 0.5]

    def test_append_is_immutable(self) -> None:
        """Append returns a new state and leaves the original unchanged."""
        s1 = make_scored(PromptCandidate("a", "t"), [make_result()])
        s2 = make_scored(PromptCandidate("b", "t"), [make_result()])
        state = make_state(s1)
        new_state = state.append(s2)
        assert len(state.scored) == 1
        assert len(new_state.scored) == 2
        assert new_state.task_description == state.task_description
