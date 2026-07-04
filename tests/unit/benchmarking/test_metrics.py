"""Unit tests for benchmarking._metrics."""

from __future__ import annotations

import pytest

from moment_to_action.benchmarking import extract_load_unload_ms
from moment_to_action.metrics import MetricsCollector, SpanType


@pytest.mark.unit
class TestExtractLoadUnloadMs:
    """Tests for extract_load_unload_ms()."""

    def test_extracts_load_and_unload_spans(self) -> None:
        """Load and unload span latencies are extracted from the report."""
        collector = MetricsCollector(session_id="test")
        with collector.start_trace():
            with collector.start_span(SpanType.MODEL_LOAD, "Model.load"):
                pass
            with collector.start_span(SpanType.MODEL_UNLOAD, "Model.unload"):
                pass

        load_ms, unload_ms = extract_load_unload_ms(collector.report())
        assert load_ms >= 0.0
        assert unload_ms >= 0.0

    def test_missing_spans_default_to_zero(self) -> None:
        """A report with no load/unload spans returns (0.0, 0.0)."""
        collector = MetricsCollector(session_id="test")
        with collector.start_trace(), collector.start_span(SpanType.STAGE, "Stage"):
            pass

        load_ms, unload_ms = extract_load_unload_ms(collector.report())
        assert (load_ms, unload_ms) == (0.0, 0.0)

    def test_name_contains_filters_by_span_name(self) -> None:
        """name_contains restricts matching to spans whose name includes the substring."""
        collector = MetricsCollector(session_id="test")
        with collector.start_trace():
            with collector.start_span(SpanType.MODEL_LOAD, "DetectorModel.load"):
                pass
            with collector.start_span(SpanType.MODEL_LOAD, "LlamaGGUFModel.load"):
                pass

        load_ms, _ = extract_load_unload_ms(collector.report(), name_contains="LlamaGGUFModel")
        assert load_ms >= 0.0

        # Only the detector span exists in a fresh report scoped to that filter.
        detector_only_collector = MetricsCollector(session_id="test2")
        with (
            detector_only_collector.start_trace(),
            detector_only_collector.start_span(SpanType.MODEL_LOAD, "DetectorModel.load"),
        ):
            pass
        load_ms2, unload_ms2 = extract_load_unload_ms(
            detector_only_collector.report(), name_contains="LlamaGGUFModel"
        )
        assert (load_ms2, unload_ms2) == (0.0, 0.0)
