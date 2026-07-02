"""Unit tests for the VLM response target."""

from __future__ import annotations

import pytest

from moment_to_action.metrics import NullMetricsCollector
from moment_to_action.prompt_tuning import (
    PromptCandidate,
    ResponseTarget,
    VLMResponseTarget,
)

from .conftest import FakeModel, make_case


@pytest.mark.unit
class TestVLMResponseTarget:
    """Tests for VLMResponseTarget.generate."""

    def test_satisfies_response_target_protocol(self) -> None:
        """VLMResponseTarget is a ResponseTarget."""
        assert isinstance(VLMResponseTarget(FakeModel()), ResponseTarget)

    def test_generate_composes_prompt_and_returns_first_response(self) -> None:
        """Generate composes the candidate prompt and returns the first response."""
        model = FakeModel(responder=lambda prompt: [f"seen[{prompt}]", "ignored"])
        target = VLMResponseTarget(model)
        candidate = PromptCandidate("You are X.", "{question}")
        case = make_case(question="Q?")

        result = target.generate(candidate, case)

        assert result == "seen[You are X.\n\nQ?]"
        assert model.prompts == ["You are X.\n\nQ?"]

    def test_generate_passes_images_and_metrics(self) -> None:
        """Generate forwards the case images and the target's metrics collector."""
        model = FakeModel()
        collector = NullMetricsCollector()
        target = VLMResponseTarget(model, metrics=collector)
        case = make_case()

        target.generate(PromptCandidate("", "{question}"), case)

        assert model.images == [["img-b64"]]
        assert model.metrics_seen == [collector]

    def test_generate_raises_when_no_responses(self) -> None:
        """Generate raises RuntimeError when the model returns no responses."""
        model = FakeModel(responder=lambda _prompt: [])
        target = VLMResponseTarget(model)
        with pytest.raises(RuntimeError, match="no responses"):
            target.generate(PromptCandidate("", "{question}"), make_case())
