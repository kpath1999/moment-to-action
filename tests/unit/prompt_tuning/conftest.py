"""Shared fakes and builders for prompt_tuning unit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs

from moment_to_action.prompt_tuning import (
    CaseResult,
    EvalCase,
    EvalReport,
    PromptCandidate,
    ScoredCandidate,
    TuningState,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from moment_to_action.metrics import MetricsCollector


class FakeModel:
    """A MultimodalModel-shaped fake that records calls and echoes prompts.

    Args:
        responder: Maps the composed prompt string to a list of responses.
            Defaults to a single echo response.
    """

    def __init__(self, responder: Callable[[str], list[str]] | None = None) -> None:
        self.responder = responder or (lambda prompt: [f"echo: {prompt}"])
        self.prompts: list[str] = []
        self.images: list[list[str]] = []
        self.metrics_seen: list[object] = []

    def prepare(
        self, inputs: tuple[str, list[str]], *, metrics: MetricsCollector | None = None
    ) -> dict[str, object]:
        """Record the prompt/images/metrics and return a prepared dict."""
        prompt, images = inputs
        self.prompts.append(prompt)
        self.images.append(images)
        self.metrics_seen.append(metrics)
        return {"prompt": prompt, "images": images}

    def run(self, prepared: object, *, metrics: MetricsCollector | None = None) -> str:
        """Return the prompt from the prepared dict."""
        del metrics
        assert isinstance(prepared, dict)
        return str(prepared["prompt"])

    def post_proc(self, raw: object, *, metrics: MetricsCollector | None = None) -> list[str]:
        """Return the responder's output for the raw prompt."""
        del metrics
        return self.responder(str(raw))


@attrs.define
class ScriptedTarget:
    """A ResponseTarget that returns canned responses and can fail some cases.

    Attributes:
        responses: Mapping of ``case_id`` to the response to return.
        fail_case_ids: Case ids for which ``generate`` raises.
        calls: Records ``(content_hash, case_id)`` for every call.
    """

    responses: dict[str, str] = attrs.Factory(dict)
    fail_case_ids: set[str] = attrs.Factory(set)
    calls: list[tuple[str, str]] = attrs.Factory(list)

    def generate(self, candidate: PromptCandidate, case: EvalCase) -> str:
        """Return the canned response for ``case`` or raise if it should fail."""
        self.calls.append((candidate.content_hash, case.case_id))
        if case.case_id in self.fail_case_ids:
            msg = f"scripted failure for {case.case_id}"
            raise RuntimeError(msg)
        return self.responses.get(case.case_id, "")


@attrs.define
class ScriptedProposer:
    """A PromptProposer that yields pre-built candidates, then raises StopTuning.

    Attributes:
        candidates: The candidates to return in order.
        seen_states: Records each state passed to ``propose``.
    """

    candidates: list[PromptCandidate] = attrs.Factory(list)
    seen_states: list[TuningState] = attrs.Factory(list)
    _index: int = attrs.field(default=0, init=False)

    def propose(self, state: TuningState) -> PromptCandidate:
        """Return the next scripted candidate or raise StopTuning when exhausted."""
        from moment_to_action.prompt_tuning import StopTuning

        self.seen_states.append(state)
        if self._index >= len(self.candidates):
            raise StopTuning
        candidate = self.candidates[self._index]
        self._index += 1
        return candidate


def make_case(
    case_id: str = "c1",
    *,
    question: str = "Is there a fight?",
    expected: str = "YES",
    keywords: tuple[str, ...] = ("yes",),
    app: str = "violence",
) -> EvalCase:
    """Build an :class:`EvalCase` with sensible defaults."""
    return EvalCase(
        case_id=case_id,
        question=question,
        images_b64=("img-b64",),
        expected_label=expected,
        recall_keywords=keywords,
        app=app,
    )


def make_result(
    case_id: str = "c1",
    *,
    score: float = 1.0,
    passed: bool = True,
    response: str = "YES a fight",
    error: str = "",
    app: str = "violence",
) -> CaseResult:
    """Build a :class:`CaseResult` with sensible defaults."""
    return CaseResult(
        case_id=case_id,
        app=app,
        question="Is there a fight?",
        expected_label="YES",
        response=response,
        score=score,
        passed=passed,
        latency_ms=1.0,
        error=error,
    )


def make_report(candidate: PromptCandidate, results: list[CaseResult]) -> EvalReport:
    """Build an :class:`EvalReport` for a candidate from result rows."""
    return EvalReport(
        candidate_label=candidate.label,
        scorer_name="keyword_recall",
        pass_threshold=0.5,
        results=tuple(results),
    )


def make_scored(candidate: PromptCandidate, results: list[CaseResult]) -> ScoredCandidate:
    """Build a :class:`ScoredCandidate` from a candidate and result rows."""
    return ScoredCandidate(candidate=candidate, report=make_report(candidate, results))


def make_state(*scored: ScoredCandidate, task: str = "detect fights") -> TuningState:
    """Build a :class:`TuningState` from scored candidates."""
    return TuningState(task_description=task, scored=tuple(scored))
