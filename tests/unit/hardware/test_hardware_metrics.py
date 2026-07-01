"""Unit tests for hardware._metrics."""

from __future__ import annotations

import pytest

from moment_to_action.hardware._metrics import InferenceMetrics, LlamaCppInferenceMetrics


@pytest.mark.unit
class TestLlamaCppInferenceMetrics:
    """Tests for LlamaCppInferenceMetrics."""

    def _make(self) -> LlamaCppInferenceMetrics:
        """Return a LlamaCppInferenceMetrics with sensible defaults."""
        return LlamaCppInferenceMetrics(
            prompt_n=10,
            prompt_ms=50.0,
            prompt_per_token_ms=5.0,
            prompt_per_second=200.0,
            predicted_n=20,
            predicted_ms=1000.0,
            predicted_per_token_ms=50.0,
            predicted_per_second=20.0,
        )

    def test_fields_stored(self) -> None:
        """All fields round-trip through construction."""
        m = self._make()
        assert m.prompt_n == 10
        assert m.prompt_ms == 50.0
        assert m.prompt_per_token_ms == 5.0
        assert m.prompt_per_second == 200.0
        assert m.predicted_n == 20
        assert m.predicted_ms == 1000.0
        assert m.predicted_per_token_ms == 50.0
        assert m.predicted_per_second == 20.0

    def test_model_dump_returns_dict(self) -> None:
        """model_dump() returns a dict with all fields."""
        m = self._make()
        d = m.model_dump()
        assert d["prompt_n"] == 10
        assert d["predicted_n"] == 20

    def test_inference_metrics_alias(self) -> None:
        """InferenceMetrics is the same type as LlamaCppInferenceMetrics."""
        m = self._make()
        assert isinstance(m, InferenceMetrics)
        assert InferenceMetrics is LlamaCppInferenceMetrics
