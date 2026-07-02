"""Persistence for the tuning trajectory.

:class:`TrajectoryStore` appends every scored candidate to a ``trajectory.jsonl``
file and keeps ``best.json`` / ``best_prompt.txt`` in sync so a run is fully
reproducible and inspectable after the fact.  Serialization is explicit (rather
than a blind :func:`attrs.asdict`) so computed fields — ``label``,
``mean_score``, ``pass_rate``, ``per_app`` — are captured for readability.

Images are never serialized; only the case ``question``/``expected_label`` that
already live on each :class:`~moment_to_action.prompt_tuning._types.CaseResult`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from pathlib import Path

    from ._types import EvalReport, PromptCandidate, ScoredCandidate

_TRAJECTORY_FILE = "trajectory.jsonl"
_BEST_JSON_FILE = "best.json"
_BEST_PROMPT_FILE = "best_prompt.txt"


def candidate_to_dict(candidate: PromptCandidate) -> dict[str, object]:
    """Serialize a candidate to a JSON-ready dict, including computed fields.

    Args:
        candidate: The candidate to serialize.

    Returns:
        A dict with the candidate's content, lineage, and derived ``label`` /
        ``content_hash``.
    """
    return {
        "label": candidate.label,
        "content_hash": candidate.content_hash,
        "generation": candidate.generation,
        "parent_id": candidate.parent_id,
        "system_prompt": candidate.system_prompt,
        "task_template": candidate.task_template,
        "rationale": candidate.rationale,
    }


def report_to_dict(report: EvalReport) -> dict[str, object]:
    """Serialize a report to a JSON-ready dict, including aggregate metrics.

    Args:
        report: The report to serialize.

    Returns:
        A dict with aggregate metrics and every per-case result.
    """
    return {
        "candidate_label": report.candidate_label,
        "scorer_name": report.scorer_name,
        "pass_threshold": report.pass_threshold,
        "num_cases": report.num_cases,
        "mean_score": report.mean_score,
        "pass_rate": report.pass_rate,
        "per_app": report.per_app(),
        "results": [attrs.asdict(r) for r in report.results],
    }


def scored_to_dict(scored: ScoredCandidate) -> dict[str, object]:
    """Serialize a scored candidate (candidate + report) to a JSON-ready dict.

    Args:
        scored: The scored candidate to serialize.

    Returns:
        A dict with ``candidate`` and ``report`` sub-objects.
    """
    return {
        "candidate": candidate_to_dict(scored.candidate),
        "report": report_to_dict(scored.report),
    }


def render_prompt(candidate: PromptCandidate) -> str:
    """Render a candidate's prompt as human-readable text.

    Args:
        candidate: The candidate to render.

    Returns:
        A labelled, multi-section string suitable for copy-paste.
    """
    return (
        f"# {candidate.label}\n"
        f"# parent: {candidate.parent_id or '(seed)'}\n"
        f"# rationale: {candidate.rationale or '(none)'}\n\n"
        f"## SYSTEM PROMPT\n{candidate.system_prompt}\n\n"
        f"## TASK TEMPLATE\n{candidate.task_template}\n"
    )


@attrs.frozen
class TrajectoryStore:
    """Writes the tuning trajectory and best prompt under a run directory.

    Attributes:
        run_dir: Directory to write ``trajectory.jsonl``, ``best.json`` and
            ``best_prompt.txt`` into (created on first write).
    """

    run_dir: Path

    def record(self, scored: ScoredCandidate) -> None:
        """Append a scored candidate to the trajectory log.

        Args:
            scored: The scored candidate to append.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(scored_to_dict(scored))
        with (self.run_dir / _TRAJECTORY_FILE).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def write_best(self, best: ScoredCandidate) -> None:
        """Write the best scored candidate as JSON and its prompt as text.

        Args:
            best: The best scored candidate to persist.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / _BEST_JSON_FILE).write_text(
            json.dumps(scored_to_dict(best), indent=2), encoding="utf-8"
        )
        (self.run_dir / _BEST_PROMPT_FILE).write_text(
            render_prompt(best.candidate), encoding="utf-8"
        )
