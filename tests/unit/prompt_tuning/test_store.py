"""Unit tests for trajectory persistence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from moment_to_action.prompt_tuning import (
    PromptCandidate,
    TrajectoryStore,
    candidate_to_dict,
    render_prompt,
    report_to_dict,
    scored_to_dict,
)

from .conftest import make_result, make_scored

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
class TestSerialization:
    """Tests for the dict/text serialization helpers."""

    def test_candidate_to_dict_includes_computed_fields(self) -> None:
        """candidate_to_dict emits label and content_hash alongside content."""
        candidate = PromptCandidate("sys", "tmpl", generation=2, parent_id="p", rationale="r")
        data = candidate_to_dict(candidate)
        assert data["label"] == candidate.label
        assert data["content_hash"] == candidate.content_hash
        assert data["system_prompt"] == "sys"
        assert data["generation"] == 2

    def test_report_to_dict_includes_metrics_and_results(self) -> None:
        """report_to_dict emits aggregates and per-case results."""
        scored = make_scored(
            PromptCandidate("s", "t"),
            [make_result("a", score=1.0), make_result("b", score=0.0, passed=False)],
        )
        data = report_to_dict(scored.report)
        assert data["num_cases"] == 2
        assert data["mean_score"] == 0.5
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 2
        assert data["results"][0]["case_id"] == "a"

    def test_scored_to_dict_is_json_serializable(self) -> None:
        """scored_to_dict round-trips through json.dumps."""
        scored = make_scored(PromptCandidate("s", "t"), [make_result()])
        text = json.dumps(scored_to_dict(scored))
        assert "candidate" in json.loads(text)
        assert "report" in json.loads(text)

    def test_render_prompt_marks_seed(self) -> None:
        """render_prompt labels a parentless candidate as the seed."""
        text = render_prompt(PromptCandidate("sys", "tmpl"))
        assert "(seed)" in text
        assert "(none)" in text
        assert "sys" in text

    def test_render_prompt_shows_parent_and_rationale(self) -> None:
        """render_prompt shows parent id and rationale when present."""
        text = render_prompt(
            PromptCandidate("sys", "tmpl", generation=1, parent_id="gen00-abc", rationale="why")
        )
        assert "gen00-abc" in text
        assert "why" in text


@pytest.mark.unit
class TestTrajectoryStore:
    """Tests for TrajectoryStore file writes."""

    def test_record_appends_one_json_line_per_call(self, tmp_path: Path) -> None:
        """Record appends a JSONL entry each time it is called."""
        store = TrajectoryStore(run_dir=tmp_path / "run")
        store.record(make_scored(PromptCandidate("a", "t"), [make_result()]))
        store.record(make_scored(PromptCandidate("b", "t"), [make_result()]))

        lines = (tmp_path / "run" / "trajectory.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert all(json.loads(line)["candidate"]["label"] for line in lines)

    def test_write_best_writes_json_and_prompt(self, tmp_path: Path) -> None:
        """write_best emits best.json and a human-readable best_prompt.txt."""
        store = TrajectoryStore(run_dir=tmp_path)
        best = make_scored(PromptCandidate("winner", "tmpl"), [make_result(score=1.0)])
        store.write_best(best)

        best_json = json.loads((tmp_path / "best.json").read_text())
        assert best_json["candidate"]["system_prompt"] == "winner"
        assert "winner" in (tmp_path / "best_prompt.txt").read_text()
