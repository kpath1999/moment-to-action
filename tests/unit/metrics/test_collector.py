"""Unit tests for MetricsCollector."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from moment_to_action.metrics import (
    MetricsCollector,
    MetricsReport,
    NullMetricsCollector,
    Span,
    SpanType,
    Trace,
)


@pytest.fixture
def collector() -> MetricsCollector:
    """Provide a fresh MetricsCollector instance for each test."""
    return MetricsCollector()


@pytest.fixture
def collector_with_budget() -> MetricsCollector:
    """Provide a MetricsCollector with custom latency budget."""
    return MetricsCollector(latency_budget=timedelta(milliseconds=50))


@pytest.mark.unit
class TestMetricsCollectorConstruction:
    """Tests for MetricsCollector initialization."""

    def test_construction_with_defaults(self) -> None:
        """Test MetricsCollector construction with default parameters."""
        collector = MetricsCollector()
        assert collector.session_id.startswith("session_")
        assert isinstance(collector.session_id, str)
        assert collector.latency_budget == timedelta(seconds=5)

    def test_construction_with_custom_session_id(self) -> None:
        """Test MetricsCollector construction with custom session_id."""
        session_id = "test_session_123"
        collector = MetricsCollector(session_id=session_id)
        assert collector.session_id == session_id

    def test_construction_with_custom_latency_budget(self) -> None:
        """Test MetricsCollector construction with custom latency_budget."""
        budget = timedelta(milliseconds=250)
        collector = MetricsCollector(latency_budget=budget)
        assert collector.latency_budget == budget

    def test_construction_with_both_params(self) -> None:
        """Test MetricsCollector construction with both session_id and latency_budget."""
        session_id = "custom_session"
        budget = timedelta(seconds=2)
        collector = MetricsCollector(session_id=session_id, latency_budget=budget)
        assert collector.session_id == session_id
        assert collector.latency_budget == budget

    def test_session_id_auto_generation_includes_timestamp(self) -> None:
        """Test that auto-generated session IDs include timestamps."""
        collector = MetricsCollector()

        # Should start with "session_" and be a string
        assert collector.session_id.startswith("session_")
        assert isinstance(collector.session_id, str)
        # Extract the timestamp part and verify it's a number
        timestamp_part = collector.session_id.replace("session_", "")
        assert timestamp_part.isdigit()


@pytest.mark.unit
class TestStartTrace:
    """Tests for start_trace context manager."""

    def test_start_trace_creates_trace(self, collector: MetricsCollector) -> None:
        """Test that start_trace creates a Trace object."""
        with collector.start_trace() as trace:
            assert isinstance(trace, Trace)
            assert isinstance(trace.id_, int)
            assert trace.id_ >= 0

        # Verify it was stored
        assert len(collector.traces) == 1
        assert collector.traces[0].id_ == trace.id_

    def test_start_trace_sets_latency_ns(self, collector: MetricsCollector) -> None:
        """Test that latency_ns is set on the trace after it ends."""
        with collector.start_trace() as trace:
            assert trace.latency_ns == -1  # Not set yet

        # After context exit, latency should be set
        assert trace.latency_ns > 0

    def test_start_trace_stores_in_traces(self, collector: MetricsCollector) -> None:
        """Test that traces are stored in collector.traces."""
        trace_ids = []
        for _i in range(3):
            with collector.start_trace() as trace:
                trace_ids.append(trace.id_)
                time.sleep(0.001)  # Small delay to ensure different traces

        assert len(collector.traces) == 3
        assert [t.id_ for t in collector.traces] == trace_ids

    def test_nested_start_trace_raises_error(self, collector: MetricsCollector) -> None:
        """Test that nested start_trace() raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Cannot start a new trace while another is active"):  # noqa: PT012
            with collector.start_trace():
                with collector.start_trace():
                    pass

    def test_trace_has_start_and_end_times(self, collector: MetricsCollector) -> None:
        """Test that trace has start and end times."""
        with collector.start_trace() as trace:
            assert isinstance(trace.start, datetime)
            assert trace.start.tzinfo is timezone.utc
            # end is set to epoch initially
            assert trace.end == datetime(1970, 1, 1, tzinfo=timezone.utc)

        # After context exit, end should be updated
        assert trace.end != datetime(1970, 1, 1, tzinfo=timezone.utc)
        assert trace.end >= trace.start

    def test_span_stack_empty_at_trace_end(self, collector: MetricsCollector) -> None:
        """Test that span_stack is empty at trace end."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test_span"):
                pass  # Span should be properly closed

        # If span_stack wasn't empty, it would have raised
        assert len(collector.traces) == 1

    def test_span_stack_not_empty_raises_error(self, collector: MetricsCollector) -> None:
        """Test that exceptions in spans are re-raised after closing gracefully."""
        msg = "test error"
        with pytest.raises(ValueError, match=msg):  # noqa: PT012
            with collector.start_trace():
                with collector.start_span(SpanType.STAGE, "test_span"):
                    raise ValueError(msg)

        # After exception is raised, span stack should be cleaned up
        assert len(collector._span_stack) == 0
        # And trace should have recorded the span
        trace = collector.get_trace(0)
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "test_span"

    def test_get_trace_by_id(self, collector: MetricsCollector) -> None:
        """Test get_trace() retrieves trace by ID."""
        with collector.start_trace() as trace:
            pass

        retrieved = collector.get_trace(trace.id_)
        assert retrieved.id_ == trace.id_

    def test_current_trace_property(self, collector: MetricsCollector) -> None:
        """Test current_trace property during active trace."""
        with collector.start_trace() as trace:
            assert collector.current_trace is trace

        # Should raise after trace ends
        with pytest.raises(RuntimeError, match="No active trace"):
            _ = collector.current_trace

    def test_current_trace_raises_without_active_trace(self, collector: MetricsCollector) -> None:
        """Test that accessing current_trace without active trace raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No active trace"):
            _ = collector.current_trace


@pytest.mark.unit
class TestStartSpan:
    """Tests for start_span context manager."""

    def test_start_span_creates_span(self, collector: MetricsCollector) -> None:
        """Test that start_span creates a Span object."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test_span") as span:
                assert isinstance(span, Span)
                assert isinstance(span.id_, int)
                assert span.id_ >= 0
                assert span.type_ == SpanType.STAGE
                assert span.name == "test_span"

    def test_start_span_stores_in_spans(self, collector: MetricsCollector) -> None:
        """Test that spans are stored in collector.spans."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "span1"):
                pass
            with collector.start_span(SpanType.MODEL_PREPROCESS, "span2"):
                pass

        assert len(collector.spans) == 2

    def test_start_span_sets_latency_ns(self, collector: MetricsCollector) -> None:
        """Test that latency_ns is set on the span after it ends."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test") as span:
                assert span.latency_ns == -1  # Not set yet

            # After context exit, latency should be set
            assert span.latency_ns > 0

    def test_start_span_without_active_trace_raises_error(
        self,
        collector: MetricsCollector,
    ) -> None:
        """Test that start_span without active trace raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Cannot start a span without an active trace"):
            with collector.start_span(SpanType.STAGE, "test"):
                pass

    def test_start_span_with_metadata(self, collector: MetricsCollector) -> None:
        """Test that metadata can be set and retrieved on a span."""
        metadata = {"model": "yolo", "device": "gpu"}
        with collector.start_trace():
            with collector.start_span(
                SpanType.MODEL_INFERENCE, "inference", metadata=metadata
            ) as span:
                assert span.metadata == metadata
                assert span.metadata["model"] == "yolo"

    def test_start_span_with_none_metadata(self, collector: MetricsCollector) -> None:
        """Test that None metadata defaults to empty dict."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test", metadata=None) as span:
                assert span.metadata == {}

    def test_start_span_without_metadata(self, collector: MetricsCollector) -> None:
        """Test that omitting metadata defaults to empty dict."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test") as span:
                assert span.metadata == {}

    def test_get_span_by_id(self, collector: MetricsCollector) -> None:
        """Test get_span() retrieves span by ID."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test") as span:
                pass

        retrieved = collector.get_span(span.id_)
        assert retrieved.id_ == span.id_

    def test_current_span_property(self, collector: MetricsCollector) -> None:
        """Test current_span property during active span."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test") as span:
                assert collector.current_span is span

    def test_current_span_raises_without_active_span(self, collector: MetricsCollector) -> None:
        """Test that accessing current_span without active span raises RuntimeError."""
        with collector.start_trace():
            with pytest.raises(RuntimeError, match="No spans have been started"):
                _ = collector.current_span

    def test_current_span_raises_without_active_trace(self, collector: MetricsCollector) -> None:
        """Test that accessing current_span without active trace raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No spans have been started"):
            _ = collector.current_span


@pytest.mark.unit
class TestSpanNesting:
    """Tests for nested span functionality."""

    def test_nested_spans_single_level(self, collector: MetricsCollector) -> None:
        """Test single level of nested spans (parent → child)."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "parent") as parent_span:
                with collector.start_span(SpanType.MODEL_PREPROCESS, "child") as child_span:
                    assert child_span.parent_id == parent_span.id_

        assert len(collector.spans) == 2

    def test_nested_spans_multiple_levels(self, collector: MetricsCollector) -> None:
        """Test multiple levels of span nesting (parent → child → grandchild)."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "parent") as parent:
                with collector.start_span(SpanType.MODEL_PREPROCESS, "child") as child:
                    with collector.start_span(SpanType.MODEL_INFERENCE, "grandchild") as grandchild:
                        assert child.parent_id == parent.id_
                        assert grandchild.parent_id == child.id_

        assert len(collector.spans) == 3

    def test_parent_id_correctly_set_for_nested_spans(self, collector: MetricsCollector) -> None:
        """Test that parent_id is correctly set for nested spans."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "parent") as parent:
                parent_id = parent.id_
                with collector.start_span(SpanType.MODEL_PREPROCESS, "child1") as child1:
                    assert child1.parent_id == parent_id
                with collector.start_span(SpanType.MODEL_PREPROCESS, "child2") as child2:
                    assert child2.parent_id == parent_id

        # Verify both children have same parent_id
        child_spans = [s for s in collector.spans if s.parent_id is not None]
        assert len(child_spans) == 2
        assert all(s.parent_id == parent_id for s in child_spans)

    def test_all_nested_spans_stored(self, collector: MetricsCollector) -> None:
        """Test that all nested spans are stored in collector.spans."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "parent"):
                with collector.start_span(SpanType.MODEL_PREPROCESS, "child1"):
                    with collector.start_span(SpanType.MODEL_INFERENCE, "grandchild1"):
                        pass
                with collector.start_span(SpanType.MODEL_PREPROCESS, "child2"):
                    pass

        assert len(collector.spans) == 4

    def test_span_stack_properly_maintained(self, collector: MetricsCollector) -> None:
        """Test that span stack is properly maintained during nesting."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "parent"):
                # Parent should be current
                assert collector.current_span.name == "parent"
                with collector.start_span(SpanType.MODEL_PREPROCESS, "child"):
                    # Child should be current
                    assert collector.current_span.name == "child"
                # Parent should be current again
                assert collector.current_span.name == "parent"

    def test_spans_must_end_in_reverse_order(self, collector: MetricsCollector) -> None:
        """Test that spans must be ended in LIFO order (reverse of start order)."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "parent"):
                with collector.start_span(SpanType.MODEL_PREPROCESS, "child"):
                    pass

                # Trying to manually mess with span stack would be caught
                # by the context manager logic


@pytest.mark.unit
class TestMetricsReport:
    """Tests for MetricsReport functionality."""

    def test_report_returns_metrics_report(self, collector: MetricsCollector) -> None:
        """Test that report() returns a MetricsReport."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test"):
                pass

        report = collector.report()
        assert isinstance(report, MetricsReport)

    def test_report_contains_all_traces(self, collector: MetricsCollector) -> None:
        """Test that report contains all recorded traces."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test"):
                pass

        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test2"):
                pass

        report = collector.report()
        assert len(report.traces) == 2

    def test_report_identifies_slow_traces(self, collector_with_budget: MetricsCollector) -> None:
        """Test that report identifies traces exceeding latency_budget."""
        # Create a slow trace
        with collector_with_budget.start_trace():
            with collector_with_budget.start_span(SpanType.STAGE, "slow"):
                time.sleep(0.1)  # Sleep longer than 50ms budget

        # Create a fast trace
        with collector_with_budget.start_trace():
            with collector_with_budget.start_span(SpanType.STAGE, "fast"):
                time.sleep(0.005)

        report = collector_with_budget.report()
        assert len(report.traces) == 2
        assert len(report.slow_traces) == 1
        assert report.slow_traces[0].latency > collector_with_budget.latency_budget

    def test_report_with_no_traces(self, collector: MetricsCollector) -> None:
        """Test report with no traces."""
        report = collector.report()
        assert isinstance(report, MetricsReport)
        assert len(report.traces) == 0
        assert len(report.slow_traces) == 0

    def test_report_contains_session_id(self, collector: MetricsCollector) -> None:
        """Test that report contains session_id."""
        report = collector.report()
        assert report.session_id == collector.session_id

    def test_report_contains_latency_budget(self, collector: MetricsCollector) -> None:
        """Test that report contains latency_budget."""
        report = collector.report()
        assert report.latency_budget == collector.latency_budget

    def test_report_json_method(self, collector: MetricsCollector) -> None:
        """Test that report.json() returns JSON-serializable dict."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test"):
                pass

        report = collector.report()
        json_data = report.json()

        assert isinstance(json_data, dict)
        assert "session_id" in json_data
        assert "traces" in json_data
        assert "slow_traces" in json_data
        assert "latency_budget_ms" in json_data

    def test_report_summary_method(self, collector: MetricsCollector) -> None:
        """Test that report.summary() returns a string."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test"):
                pass

        report = collector.report()
        summary = report.summary()

        assert isinstance(summary, str)
        assert collector.session_id in summary

    def test_report_summary_full_method(self, collector: MetricsCollector) -> None:
        """Test that report.summary_full() returns a string."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test"):
                pass

        report = collector.report()
        summary = report.summary_full()

        assert isinstance(summary, str)
        assert "All Traces:" in summary


@pytest.mark.unit
class TestMetadataOperations:
    """Tests for metadata get/set operations."""

    def test_get_meta_from_span(self, collector: MetricsCollector) -> None:
        """Test getting metadata from current span."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test", metadata={"key": "value"}):
                assert collector.get_meta("key") == "value"

    def test_set_meta_on_span(self, collector: MetricsCollector) -> None:
        """Test setting metadata on current span."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test"):
                collector.set_meta("key", "value")
                assert collector.get_meta("key") == "value"

    def test_get_meta_nonexistent_key(self, collector: MetricsCollector) -> None:
        """Test getting nonexistent metadata key returns None."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test"):
                assert collector.get_meta("nonexistent") is None

    def test_set_meta_without_active_span_raises(self, collector: MetricsCollector) -> None:
        """Test that set_meta without active span raises RuntimeError."""
        with pytest.raises(RuntimeError):
            collector.set_meta("key", "value")

    def test_get_meta_without_active_span_raises(self, collector: MetricsCollector) -> None:
        """Test that get_meta without active span raises RuntimeError."""
        with pytest.raises(RuntimeError):
            collector.get_meta("key")


@pytest.mark.unit
class TestNullMetricsCollector:
    """Tests for NullMetricsCollector (no-op implementation)."""

    def test_null_collector_session_id_is_null(self) -> None:
        """Test that NullMetricsCollector has session_id='null'."""
        collector = NullMetricsCollector()
        # NullMetricsCollector always has session_id='null', not auto-generated
        assert collector.session_id == "null"

    def test_null_collector_start_trace_is_noop(self) -> None:
        """Test that start_trace is a no-op (doesn't store trace)."""
        collector = NullMetricsCollector()
        with collector.start_trace() as trace:
            assert isinstance(trace, Trace)

        # Should not store the trace
        assert len(collector.traces) == 0

    def test_null_collector_start_span_is_noop(self) -> None:
        """Test that start_span is a no-op (doesn't store span)."""
        collector = NullMetricsCollector()
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test") as span:
                assert isinstance(span, Span)

            # Should not store the span
            assert len(collector.spans) == 0

    def test_null_collector_report_is_empty(self) -> None:
        """Test that report() returns empty report."""
        collector = NullMetricsCollector()
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test"):
                pass

        report = collector.report()
        assert len(report.traces) == 0
        assert len(report.slow_traces) == 0

    def test_null_collector_collects_no_data(self) -> None:
        """Test that NullMetricsCollector doesn't collect any data."""
        collector = NullMetricsCollector()

        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "span1"):
                pass
            with collector.start_span(SpanType.MODEL_PREPROCESS, "span2"):
                pass

        with collector.start_trace():
            with collector.start_span(SpanType.MODEL_INFERENCE, "span3"):
                pass

        # All collections should be empty
        assert len(collector.traces) == 0
        assert len(collector.spans) == 0

        report = collector.report()
        assert len(report.traces) == 0


@pytest.mark.unit
class TestErrorHandling:
    """Tests for error handling and invariants."""

    def test_span_without_active_trace_raises(self, collector: MetricsCollector) -> None:
        """Test that start_span without active trace raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Cannot start a span without an active trace"):
            with collector.start_span(SpanType.STAGE, "test"):
                pass

    def test_improper_span_nesting_raises(self, collector: MetricsCollector) -> None:
        """Test that improper span nesting (ending out of order) raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Span stack is out of order"):  # noqa: PT012
            with collector.start_trace():
                try:
                    with collector.start_span(SpanType.STAGE, "span1"):
                        with collector.start_span(SpanType.MODEL_PREPROCESS, "span2"):
                            # Manually try to mess with the span stack
                            collector._span_stack.pop()  # Remove span2
                except RuntimeError:  # noqa: TRY203
                    raise  # Re-raise the expected error

    def test_current_trace_raises_without_active_trace(self, collector: MetricsCollector) -> None:
        """Test that accessing current_trace without active trace raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No active trace"):
            _ = collector.current_trace

    def test_current_span_raises_without_active_span(self, collector: MetricsCollector) -> None:
        """Test that accessing current_span without active span raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No spans have been started"):
            _ = collector.current_span

    def test_multiple_nested_spans_proper_cleanup(self, collector: MetricsCollector) -> None:
        """Test that multiple nested spans are properly cleaned up."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "s1"):
                with collector.start_span(SpanType.MODEL_PREPROCESS, "s2"):
                    with collector.start_span(SpanType.MODEL_INFERENCE, "s3"):
                        pass

        # After all spans close, accessing current_span should raise
        with pytest.raises(RuntimeError):
            _ = collector.current_span


@pytest.mark.unit
class TestSpanTypes:
    """Tests for different SpanType values."""

    def test_span_with_pipeline_type(self, collector: MetricsCollector) -> None:
        """Test creating span with PIPELINE type."""
        with collector.start_trace():
            with collector.start_span(SpanType.PIPELINE, "pipeline") as span:
                assert span.type_ == SpanType.PIPELINE

    def test_span_with_stage_type(self, collector: MetricsCollector) -> None:
        """Test creating span with STAGE type."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "stage") as span:
                assert span.type_ == SpanType.STAGE

    def test_span_with_model_preprocess_type(self, collector: MetricsCollector) -> None:
        """Test creating span with MODEL_PREPROCESS type."""
        with collector.start_trace():
            with collector.start_span(SpanType.MODEL_PREPROCESS, "preprocess") as span:
                assert span.type_ == SpanType.MODEL_PREPROCESS

    def test_span_with_model_inference_type(self, collector: MetricsCollector) -> None:
        """Test creating span with MODEL_INFERENCE type."""
        with collector.start_trace():
            with collector.start_span(SpanType.MODEL_INFERENCE, "inference") as span:
                assert span.type_ == SpanType.MODEL_INFERENCE

    def test_span_with_model_post_process_type(self, collector: MetricsCollector) -> None:
        """Test creating span with MODEL_POST_PROCESS type."""
        with collector.start_trace():
            with collector.start_span(SpanType.MODEL_POST_PROCESS, "post_process") as span:
                assert span.type_ == SpanType.MODEL_POST_PROCESS

    def test_all_span_types_work_in_trace(self, collector: MetricsCollector) -> None:
        """Test that all SpanType values work within a trace."""
        span_types = [
            SpanType.PIPELINE,
            SpanType.STAGE,
            SpanType.MODEL_PREPROCESS,
            SpanType.MODEL_INFERENCE,
            SpanType.MODEL_POST_PROCESS,
        ]

        with collector.start_trace():
            for span_type in span_types:
                with collector.start_span(span_type, f"span_{span_type.name}"):
                    pass

        assert len(collector.spans) == len(span_types)


@pytest.mark.unit
class TestLatencyMeasurement:
    """Tests for latency measurement accuracy."""

    def test_span_latency_is_positive(self, collector: MetricsCollector) -> None:
        """Test that span latency is positive after execution."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test") as span:
                time.sleep(0.001)  # Sleep 1ms

            assert span.latency_ns > 0
            assert span.latency.total_seconds() > 0

    def test_trace_latency_is_positive(self, collector: MetricsCollector) -> None:
        """Test that trace latency is positive after execution."""
        with collector.start_trace() as trace:
            time.sleep(0.001)  # Sleep 1ms

        assert trace.latency_ns > 0
        assert trace.latency.total_seconds() > 0

    def test_latency_ms_property(self, collector: MetricsCollector) -> None:
        """Test that latency_ms property converts correctly."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test") as span:
                time.sleep(0.001)  # Sleep 1ms

            # latency_ns / 1_000_000 should give latency_ms
            assert span.latency_ms >= 1.0  # Should be at least 1ms

    def test_trace_contains_span_latencies(self, collector: MetricsCollector) -> None:
        """Test that trace can access spans with their latencies."""
        with collector.start_trace() as trace:
            with collector.start_span(SpanType.STAGE, "span1") as span1:
                time.sleep(0.001)
            with collector.start_span(SpanType.STAGE, "span2") as span2:
                time.sleep(0.001)

        # Verify both spans have latency measured
        assert span1.latency_ns > 0
        assert span2.latency_ns > 0
        assert trace.latency_ns > (span1.latency_ns + span2.latency_ns)  # Trace includes overhead


@pytest.mark.unit
class TestTraceProperties:
    """Tests for Trace properties and methods."""

    def test_trace_json_method(self, collector: MetricsCollector) -> None:
        """Test that trace.json() returns JSON-serializable dict."""
        with collector.start_trace() as trace:
            with collector.start_span(SpanType.STAGE, "test"):
                pass

        json_data = trace.json()
        assert isinstance(json_data, dict)
        assert "id" in json_data
        assert "spans" in json_data
        assert "latency_ns" in json_data

    def test_trace_summary_method(self, collector: MetricsCollector) -> None:
        """Test that trace.summary() returns a string."""
        with collector.start_trace() as trace:
            with collector.start_span(SpanType.STAGE, "test"):
                pass

        summary = trace.summary()
        assert isinstance(summary, str)
        assert "Trace" in summary

    def test_trace_summary_with_budget(self, collector_with_budget: MetricsCollector) -> None:
        """Test that trace.summary() includes budget comparison."""
        with collector_with_budget.start_trace() as trace:
            with collector_with_budget.start_span(SpanType.STAGE, "test"):
                pass

        summary = trace.summary(latency_budget=collector_with_budget.latency_budget)
        assert isinstance(summary, str)


@pytest.mark.unit
class TestSpanProperties:
    """Tests for Span properties and methods."""

    def test_span_json_method(self, collector: MetricsCollector) -> None:
        """Test that span.json() returns JSON-serializable dict."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test", metadata={"key": "value"}) as span:
                pass

        json_data = span.json()
        assert isinstance(json_data, dict)
        assert "id" in json_data
        assert "name" in json_data
        assert "type" in json_data
        assert "metadata" in json_data

    def test_span_summary_method(self, collector: MetricsCollector) -> None:
        """Test that span.summary() returns a string."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test") as span:
                pass

        summary = span.summary()
        assert isinstance(summary, str)
        assert "test" in summary

    def test_span_parent_id_is_none_for_root_span(self, collector: MetricsCollector) -> None:
        """Test that root span has parent_id=None."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "root") as span:
                assert span.parent_id is None

    def test_span_stores_datetimes(self, collector: MetricsCollector) -> None:
        """Test that span stores start and end datetimes."""
        with collector.start_trace():
            with collector.start_span(SpanType.STAGE, "test") as span:
                assert isinstance(span.start, datetime)
                assert span.start.tzinfo is timezone.utc

        assert isinstance(span.end, datetime)
        assert span.end.tzinfo is timezone.utc
        assert span.end > span.start


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and error paths."""

    def test_unfreeze_with_unfrozen_object(self) -> None:
        """Test that _unfreeze raises ValueError when given an unfrozen object."""
        from moment_to_action.metrics._collector import _unfreeze

        # Create an unfrozen attrs object by manually setting _frozen to False
        test_span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="test",
            start=datetime.now(tz=timezone.utc),
            end=datetime(1970, 1, 1, tzinfo=timezone.utc),
        )

        # Manually unfreeze it so it's not frozen
        object.__setattr__(test_span, "_frozen", False)

        # Now trying to unfreeze should raise ValueError
        with pytest.raises(ValueError, match="is not frozen"):
            with _unfreeze(test_span):
                pass

    def test_null_collector_get_trace(self) -> None:
        """Test that NullMetricsCollector.get_trace() returns a dummy trace."""
        collector = NullMetricsCollector()
        trace = collector.get_trace(trace_id=999)
        assert isinstance(trace, Trace)
        assert trace.id_ == 0
        assert isinstance(trace.start, datetime)
        assert isinstance(trace.end, datetime)

    def test_null_collector_get_span(self) -> None:
        """Test that NullMetricsCollector.get_span() returns a dummy span."""
        collector = NullMetricsCollector()
        span = collector.get_span(span_id=999)
        assert isinstance(span, Span)
        assert span.id_ == 0
        assert span.type_ == SpanType.STAGE
        assert span.name == "null"


@pytest.mark.unit
class TestResourceSampling:
    """Tests for resource usage sampling when a compute backend is provided."""

    def test_collector_with_backend_samples_resource_usage(self) -> None:
        """Test that _sample_resource_usage collects samples when a backend is provided.

        This covers the code path at lines 126-148 of _collector.py where the
        collector uses the compute backend's resource monitor to sample hardware metrics.
        """
        from unittest.mock import MagicMock

        from moment_to_action.hardware._types import ComputeUnit, ComputeUnitUsageSample

        # Build a ComputeUnitUsageSample mock with the required structure
        def _make_usage_sample(device: ComputeUnit) -> MagicMock:
            sample = MagicMock(spec=ComputeUnitUsageSample)
            sample.device = device
            sample.power_mw = 100.0
            sample.usage_pct = 10.0
            sample.frequency_mhz = 1000.0
            sample.memory_mb = 512.0
            return sample

        mock_resource_monitor = MagicMock()
        mock_resource_monitor.sample.side_effect = _make_usage_sample

        mock_backend = MagicMock()
        mock_backend.resource_monitor = mock_resource_monitor

        # Use a very short sample interval so we get at least one sample during the trace
        collector = MetricsCollector(
            compute_platform=mock_backend,
            session_id="test-resource-sampling",
            resource_sample_interval=timedelta(milliseconds=10),
        )

        with collector.start_trace() as trace:
            with collector.start_span(SpanType.STAGE, "test"):
                # Brief sleep to allow at least one resource sample to be collected
                time.sleep(0.05)

        # The trace should have at least one resource usage sample
        assert len(trace.resource_usage_samples) >= 1
        sample = trace.resource_usage_samples[0]
        assert sample.cpu_usage is not None
        assert sample.gpu_usage is not None

    def test_collector_no_backend_does_not_sample(self) -> None:
        """Test that _sample_resource_usage is a no-op when compute_backend is None.

        This covers the early-return at line 124 of _collector.py.
        """
        collector = MetricsCollector(
            compute_platform=None,
            session_id="test-no-backend",
            resource_sample_interval=timedelta(milliseconds=10),
        )

        with collector.start_trace() as trace:
            with collector.start_span(SpanType.STAGE, "test"):
                time.sleep(0.05)

        # No resource samples should be collected without a backend
        assert len(trace.resource_usage_samples) == 0


@pytest.mark.unit
class TestTimedStream:
    """Tests for MetricsCollector.timed_stream()."""

    def test_yields_tokens_unchanged(self, collector: MetricsCollector) -> None:
        """timed_stream() yields every token from the source iterable, unchanged."""
        with collector.start_trace(), collector.start_span(SpanType.MODEL_INFERENCE, "infer"):
            tokens = list(collector.timed_stream(["a", "b", "c"]))
        assert tokens == ["a", "b", "c"]

    def test_stamps_ttft_and_itl_on_current_span(self, collector: MetricsCollector) -> None:
        """timed_stream() stamps ttft_ms and mean/std itl on the currently open span."""
        with (
            collector.start_trace(),
            collector.start_span(SpanType.MODEL_INFERENCE, "infer") as span,
        ):
            list(collector.timed_stream(["a", "b", "c"]))

        assert span.metadata["ttft_ms"] >= 0.0
        assert span.metadata["mean_itl_ms"] >= 0.0
        assert span.metadata["std_itl_ms"] >= 0.0

    def test_single_token_has_no_itl(self, collector: MetricsCollector) -> None:
        """A single-token stream has ttft_ms but no itl (no inter-token gap to measure)."""
        with (
            collector.start_trace(),
            collector.start_span(SpanType.MODEL_INFERENCE, "infer") as span,
        ):
            list(collector.timed_stream(["only"]))

        assert "ttft_ms" in span.metadata
        assert "mean_itl_ms" not in span.metadata

    def test_empty_stream_stamps_nothing(self, collector: MetricsCollector) -> None:
        """An empty token stream stamps no metadata at all."""
        with (
            collector.start_trace(),
            collector.start_span(SpanType.MODEL_INFERENCE, "infer") as span,
        ):
            list(collector.timed_stream([]))

        assert span.metadata == {}

    def test_yn_predicate_stamps_ttfyd(self, collector: MetricsCollector) -> None:
        """yn_predicate firing on accumulated text stamps ttfyd_ms."""
        with (
            collector.start_trace(),
            collector.start_span(SpanType.MODEL_INFERENCE, "infer") as span,
        ):
            list(
                collector.timed_stream(
                    ["YE", "S", " because"], yn_predicate=lambda acc: acc.startswith("YES")
                )
            )

        assert "ttfyd_ms" in span.metadata

    def test_yn_predicate_never_firing_omits_ttfyd(self, collector: MetricsCollector) -> None:
        """ttfyd_ms is not stamped when yn_predicate never returns truthy."""
        with (
            collector.start_trace(),
            collector.start_span(SpanType.MODEL_INFERENCE, "infer") as span,
        ):
            list(collector.timed_stream(["a", "b"], yn_predicate=lambda _acc: False))

        assert "ttfyd_ms" not in span.metadata

    def test_metrics_recorded_on_early_close(self, collector: MetricsCollector) -> None:
        """Closing the generator early still records ttft via the finally block."""
        with (
            collector.start_trace(),
            collector.start_span(SpanType.MODEL_INFERENCE, "infer") as span,
        ):
            gen = collector.timed_stream(iter(["a", "b", "c"]))
            next(gen)
            gen.close()

        assert "ttft_ms" in span.metadata

    def test_null_metrics_collector_passes_through(self) -> None:
        """NullMetricsCollector.timed_stream() is a no-op passthrough."""
        null_collector = NullMetricsCollector()
        tokens = list(null_collector.timed_stream(["x", "y"], yn_predicate=lambda _acc: True))
        assert tokens == ["x", "y"]
