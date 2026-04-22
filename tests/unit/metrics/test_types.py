"""Unit tests for metrics types."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import attrs
import pytest

from moment_to_action.hardware._types import ComputeUnit, ComputeUnitUsageSample
from moment_to_action.metrics._types import (
    MemoryUsageSample,
    MetricsReport,
    ResourceUsageSample,
    Span,
    SpanType,
    Trace,
)


@pytest.fixture
def utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(tz=UTC)


@pytest.fixture
def utc_past(utc_now: datetime) -> datetime:
    """Return a UTC datetime 100ms in the past."""
    return utc_now - timedelta(milliseconds=100)


@pytest.mark.unit
class TestSpanType:
    """Tests for SpanType enum."""

    def test_spantype_pipeline_member(self) -> None:
        """Test SpanType.PIPELINE member."""
        assert hasattr(SpanType, "PIPELINE")
        assert isinstance(SpanType.PIPELINE, SpanType)

    def test_spantype_stage_member(self) -> None:
        """Test SpanType.STAGE member."""
        assert hasattr(SpanType, "STAGE")
        assert isinstance(SpanType.STAGE, SpanType)

    def test_spantype_preprocess_member(self) -> None:
        """Test SpanType.PREPROCESS member."""
        assert hasattr(SpanType, "PREPROCESS")
        assert isinstance(SpanType.PREPROCESS, SpanType)

    def test_spantype_model_inference_member(self) -> None:
        """Test SpanType.MODEL_INFERENCE member."""
        assert hasattr(SpanType, "MODEL_INFERENCE")
        assert isinstance(SpanType.MODEL_INFERENCE, SpanType)

    def test_spantype_all_members(self) -> None:
        """Test that all expected SpanType members exist."""
        members = [member.name for member in SpanType]
        assert "PIPELINE" in members
        assert "STAGE" in members
        assert "PREPROCESS" in members
        assert "MODEL_INFERENCE" in members
        assert "POSTPROCESS" in members
        assert len(members) == 5


@pytest.mark.unit
class TestSpan:
    """Tests for Span attrs class."""

    def test_span_construction_basic(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test basic Span construction."""
        span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="TestStage",
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
            metadata={"key": "value"},
        )
        assert span.id_ == 1
        assert span.parent_id is None
        assert span.type_ == SpanType.STAGE
        assert span.name == "TestStage"
        assert span.start == utc_past
        assert span.end == utc_now
        assert span.latency_ns == 100_000_000
        assert span.metadata == {"key": "value"}

    def test_span_construction_with_parent_id(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Span construction with a parent span ID."""
        span = Span(
            id_=2,
            parent_id=1,
            type_=SpanType.MODEL_INFERENCE,
            name="ModelInference",
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        assert span.id_ == 2
        assert span.parent_id == 1
        assert span.type_ == SpanType.MODEL_INFERENCE

    def test_span_construction_default_metadata(
        self, utc_past: datetime, utc_now: datetime
    ) -> None:
        """Test Span construction with default empty metadata."""
        span = Span(
            id_=3,
            parent_id=None,
            type_=SpanType.PREPROCESS,
            name="Preprocessing",
            start=utc_past,
            end=utc_now,
        )
        assert span.metadata == {}

    def test_span_latency_property(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Span latency property returns a timedelta."""
        span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="TestSpan",
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
        )
        latency = span.latency
        assert isinstance(latency, timedelta)
        assert latency.total_seconds() == 0.1

    def test_span_latency_ms_property(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Span latency_ms property returns milliseconds as a float."""
        span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="TestSpan",
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
        )
        latency_ms = span.latency_ms
        assert isinstance(latency_ms, float)
        assert latency_ms == 100.0

    def test_span_latency_ms_with_smaller_duration(
        self, utc_past: datetime, utc_now: datetime
    ) -> None:
        """Test Span latency_ms with smaller durations."""
        span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="QuickSpan",
            start=utc_past,
            end=utc_now,
            latency_ns=5_000_000,
        )
        assert span.latency_ms == 5.0

    def test_span_summary_method(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Span summary() method returns a string."""
        span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="TestSpan",
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
            metadata={"key": "value"},
        )
        summary = span.summary()
        assert isinstance(summary, str)
        assert "STAGE" in summary
        assert "TestSpan" in summary
        assert "100.00ms" in summary
        assert "key" in summary or "value" in summary

    def test_span_summary_with_different_types(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Span summary() with different span types."""
        for span_type in SpanType:
            span = Span(
                id_=1,
                parent_id=None,
                type_=span_type,
                name="TestSpan",
                start=utc_past,
                end=utc_now,
                latency_ns=50_000_000,
            )
            summary = span.summary()
            assert span_type.name in summary

    def test_span_summary_rich_method(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Span summary_rich() method with rich formatting."""
        span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="TestStage",
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        summary = span.summary_rich()
        assert isinstance(summary, str)
        assert "TestStage" in summary
        assert "50.00ms" in summary
        assert "[cyan]" in summary or "STAGE" in summary

    def test_span_json_method(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Span json() method returns a dict with correct keys."""
        span = Span(
            id_=42,
            parent_id=1,
            type_=SpanType.MODEL_INFERENCE,
            name="Inference",
            start=utc_past,
            end=utc_now,
            latency_ns=200_000_000,
            metadata={"model": "yolo"},
        )
        json_dict = span.json()
        assert isinstance(json_dict, dict)
        assert json_dict["id"] == 42
        assert json_dict["parent_id"] == 1
        assert json_dict["type"] == "MODEL_INFERENCE"
        assert json_dict["name"] == "Inference"
        assert json_dict["start"] == utc_past.isoformat()
        assert json_dict["end"] == utc_now.isoformat()
        assert json_dict["latency_ns"] == 200_000_000
        assert json_dict["metadata"] == {"model": "yolo"}

    def test_span_json_with_none_parent(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Span json() with None parent_id."""
        span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.PIPELINE,
            name="Pipeline",
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
        )
        json_dict = span.json()
        assert json_dict["parent_id"] is None

    def test_span_is_frozen(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test that spans are frozen (immutable after creation)."""
        span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="TestSpan",
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
        )
        with pytest.raises(AttributeError, match="frozen"):
            span.name = "NewName"

    def test_span_is_attrs_class(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test that Span is an attrs class."""
        assert attrs.has(Span)
        span = Span(  # type: ignore[call-arg]
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="Test",
            start=utc_past,
            end=utc_now,
        )
        as_dict = attrs.asdict(span)
        assert "id_" in as_dict
        assert "type_" in as_dict


@pytest.mark.unit
class TestTrace:
    """Tests for Trace attrs class."""

    def test_trace_construction_basic(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test basic Trace construction."""
        trace = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
        )
        assert trace.id_ == 1
        assert trace.start == utc_past
        assert trace.end == utc_now
        assert trace.latency_ns == 100_000_000
        assert trace.spans == []

    def test_trace_construction_with_spans(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Trace construction with spans."""
        span1 = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.PIPELINE,
            name="Pipeline",
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
        )
        span2 = Span(
            id_=2,
            parent_id=1,
            type_=SpanType.STAGE,
            name="Stage",
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        trace = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
            spans=[span1, span2],
        )
        assert len(trace.spans) == 2
        assert trace.spans[0].id_ == 1
        assert trace.spans[1].id_ == 2

    def test_trace_latency_property(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Trace latency property returns a timedelta."""
        trace = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
        )
        latency = trace.latency
        assert isinstance(latency, timedelta)
        assert latency.total_seconds() == 0.1

    def test_trace_latency_ms_property(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Trace latency_ms property returns milliseconds as a float."""
        trace = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=150_000_000,
        )
        latency_ms = trace.latency_ms
        assert isinstance(latency_ms, float)
        assert latency_ms == 150.0

    def test_trace_summary_without_budget(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Trace summary() method without latency_budget."""
        span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="TestStage",
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        trace = Trace(
            id_=42,
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
            spans=[span],
        )
        summary = trace.summary()
        assert isinstance(summary, str)
        assert "Trace 42" in summary
        assert "100.00ms" in summary
        assert "TestStage" in summary
        assert "Within latency budget: N/A" in summary

    def test_trace_summary_with_budget_within(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Trace summary() with latency_budget when within budget."""
        span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="TestStage",
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        trace = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
            spans=[span],
        )
        budget = timedelta(milliseconds=100)
        summary = trace.summary(latency_budget=budget)
        assert isinstance(summary, str)
        assert "✅" in summary

    def test_trace_summary_with_budget_exceeded(
        self, utc_past: datetime, utc_now: datetime
    ) -> None:
        """Test Trace summary() with latency_budget when exceeded."""
        span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="TestStage",
            start=utc_past,
            end=utc_now,
            latency_ns=200_000_000,
        )
        trace = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=200_000_000,
            spans=[span],
        )
        budget = timedelta(milliseconds=100)
        summary = trace.summary(latency_budget=budget)
        assert "❌" in summary

    def test_trace_summary_with_nested_spans(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Trace summary() with nested spans."""
        parent_span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="ParentStage",
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
        )
        child_span = Span(
            id_=2,
            parent_id=1,
            type_=SpanType.MODEL_INFERENCE,
            name="ChildInference",
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        trace = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
            spans=[parent_span, child_span],
        )
        summary = trace.summary()
        assert "ParentStage" in summary
        assert "ChildInference" in summary

    def test_trace_summary_rich_method(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Trace summary_rich() method with rich formatting."""
        parent_span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="ParentStage",
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
        )
        child_span = Span(
            id_=2,
            parent_id=1,
            type_=SpanType.MODEL_INFERENCE,
            name="ChildInference",
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        trace = Trace(
            id_=42,
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
            spans=[parent_span, child_span],
        )
        summary = trace.summary_rich()
        assert isinstance(summary, str)
        assert "Trace 42" in summary
        assert "ParentStage" in summary
        assert "ChildInference" in summary
        assert "[bold]" in summary or "[green]" in summary

    def test_trace_json_method(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test Trace json() method returns a dict with correct keys."""
        span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="TestStage",
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        trace = Trace(
            id_=99,
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
            spans=[span],
        )
        json_dict = trace.json()
        assert isinstance(json_dict, dict)
        assert json_dict["id"] == 99
        assert json_dict["start"] == utc_past.isoformat()
        assert json_dict["end"] == utc_now.isoformat()
        assert json_dict["latency_ns"] == 100_000_000
        assert isinstance(json_dict["spans"], list)
        assert len(json_dict["spans"]) == 1

    def test_trace_json_recursively_calls_span_json(
        self, utc_past: datetime, utc_now: datetime
    ) -> None:
        """Test that Trace json() recursively calls json() on child spans."""
        span1 = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.PIPELINE,
            name="Pipeline",
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
            metadata={"key1": "value1"},
        )
        span2 = Span(
            id_=2,
            parent_id=1,
            type_=SpanType.STAGE,
            name="Stage",
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
            metadata={"key2": "value2"},
        )
        trace = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
            spans=[span1, span2],
        )
        json_dict = trace.json()
        assert len(json_dict["spans"]) == 2
        assert json_dict["spans"][0]["id"] == 1
        assert json_dict["spans"][0]["metadata"] == {"key1": "value1"}
        assert json_dict["spans"][1]["id"] == 2
        assert json_dict["spans"][1]["metadata"] == {"key2": "value2"}

    def test_trace_is_frozen(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test that traces are frozen (immutable after creation)."""
        trace = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
        )
        with pytest.raises(AttributeError, match="frozen"):
            trace.id_ = 2

    def test_trace_is_attrs_class(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test that Trace is an attrs class."""
        assert attrs.has(Trace)
        trace = Trace(  # type: ignore[call-arg]
            id_=1,
            start=utc_past,
            end=utc_now,
        )
        as_dict = attrs.asdict(trace)
        assert "id_" in as_dict
        assert "spans" in as_dict


@pytest.mark.unit
class TestMetricsReport:
    """Tests for MetricsReport attrs frozen dataclass."""

    def test_metricsreport_construction_basic(self) -> None:
        """Test basic MetricsReport construction."""
        report = MetricsReport(
            session_id="session_123",
            latency_budget=timedelta(milliseconds=100),
            traces=[],
            slow_traces=[],
        )
        assert report.session_id == "session_123"
        assert report.latency_budget == timedelta(milliseconds=100)
        assert report.traces == []
        assert report.slow_traces == []

    def test_metricsreport_construction_with_traces(
        self, utc_past: datetime, utc_now: datetime
    ) -> None:
        """Test MetricsReport construction with traces."""
        trace1 = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        trace2 = Trace(
            id_=2,
            start=utc_past,
            end=utc_now,
            latency_ns=200_000_000,
        )
        report = MetricsReport(
            session_id="test_session",
            latency_budget=timedelta(milliseconds=100),
            traces=[trace1, trace2],
            slow_traces=[trace2],
        )
        assert len(report.traces) == 2
        assert len(report.slow_traces) == 1
        assert report.slow_traces[0].id_ == 2

    def test_metricsreport_summary_method(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test MetricsReport summary() method returns a string."""
        trace1 = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        trace2_slow = Trace(
            id_=2,
            start=utc_past,
            end=utc_now,
            latency_ns=200_000_000,
        )
        report = MetricsReport(
            session_id="test_session",
            latency_budget=timedelta(milliseconds=100),
            traces=[trace1, trace2_slow],
            slow_traces=[trace2_slow],
        )
        summary = report.summary()
        assert isinstance(summary, str)
        assert "test_session" in summary
        assert "100.00ms" in summary
        assert "Total traces: 2" in summary
        assert "Slow traces: 1" in summary

    def test_metricsreport_summary_with_no_slow_traces(
        self, utc_past: datetime, utc_now: datetime
    ) -> None:
        """Test MetricsReport summary() when no slow traces."""
        trace = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        report = MetricsReport(
            session_id="test",
            latency_budget=timedelta(milliseconds=100),
            traces=[trace],
            slow_traces=[],
        )
        summary = report.summary()
        assert "Slow traces: 0" in summary

    def test_metricsreport_summary_full_method(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test MetricsReport summary_full() method returns a string."""
        trace1 = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        trace2 = Trace(
            id_=2,
            start=utc_past,
            end=utc_now,
            latency_ns=200_000_000,
        )
        report = MetricsReport(
            session_id="full_test",
            latency_budget=timedelta(milliseconds=100),
            traces=[trace1, trace2],
            slow_traces=[trace2],
        )
        summary_full = report.summary_full()
        assert isinstance(summary_full, str)
        assert "full_test" in summary_full
        assert "All Traces:" in summary_full
        assert "Total traces: 2" in summary_full

    def test_metricsreport_summary_rich_method(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test MetricsReport summary_rich() method with rich formatting."""
        trace = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        report = MetricsReport(
            session_id="rich_test",
            latency_budget=timedelta(milliseconds=100),
            traces=[trace],
            slow_traces=[trace],
        )
        summary = report.summary_rich()
        assert isinstance(summary, str)
        assert "rich_test" in summary
        assert "[bold]" in summary or "[green]" in summary

    def test_metricsreport_summary_full_rich_method(
        self, utc_past: datetime, utc_now: datetime
    ) -> None:
        """Test MetricsReport summary_full_rich() method with rich formatting."""
        trace1 = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        trace2 = Trace(
            id_=2,
            start=utc_past,
            end=utc_now,
            latency_ns=200_000_000,
        )
        report = MetricsReport(
            session_id="full_rich_test",
            latency_budget=timedelta(milliseconds=100),
            traces=[trace1, trace2],
            slow_traces=[trace2],
        )
        summary_full = report.summary_full_rich()
        assert isinstance(summary_full, str)
        assert "full_rich_test" in summary_full
        assert "[bold]All Traces:[/bold]" in summary_full
        assert "[bold]" in summary_full or "[green]" in summary_full

    def test_metricsreport_json_method(self, utc_past: datetime, utc_now: datetime) -> None:
        """Test MetricsReport json() method returns a dict with correct keys."""
        span = Span(
            id_=1,
            parent_id=None,
            type_=SpanType.STAGE,
            name="TestStage",
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        trace = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
            spans=[span],
        )
        report = MetricsReport(
            session_id="json_test",
            latency_budget=timedelta(milliseconds=100),
            traces=[trace],
            slow_traces=[],
        )
        json_dict = report.json()
        assert isinstance(json_dict, dict)
        assert json_dict["session_id"] == "json_test"
        assert json_dict["latency_budget_ms"] == 100.0
        assert isinstance(json_dict["traces"], list)
        assert isinstance(json_dict["slow_traces"], list)

    def test_metricsreport_json_recursively_calls_trace_json(
        self, utc_past: datetime, utc_now: datetime
    ) -> None:
        """Test that MetricsReport json() recursively calls json() on all traces."""
        trace1 = Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=50_000_000,
        )
        trace2_slow = Trace(
            id_=2,
            start=utc_past,
            end=utc_now,
            latency_ns=200_000_000,
        )
        report = MetricsReport(
            session_id="recursive_test",
            latency_budget=timedelta(milliseconds=100),
            traces=[trace1, trace2_slow],
            slow_traces=[trace2_slow],
        )
        json_dict = report.json()
        assert len(json_dict["traces"]) == 2
        assert json_dict["traces"][0]["id"] == 1
        assert json_dict["traces"][1]["id"] == 2
        assert len(json_dict["slow_traces"]) == 1
        assert json_dict["slow_traces"][0]["id"] == 2

    def test_metricsreport_is_frozen(self) -> None:
        """Test that MetricsReport is frozen (immutable after creation)."""
        report = MetricsReport(
            session_id="frozen_test",
            latency_budget=timedelta(milliseconds=100),
            traces=[],
            slow_traces=[],
        )
        with pytest.raises(AttributeError):
            report.session_id = "new_session"  # type: ignore[misc]

    def test_metricsreport_is_attrs_frozen(self) -> None:
        """Test that MetricsReport is an attrs frozen class."""
        assert attrs.has(MetricsReport)
        report = MetricsReport(  # type: ignore[call-arg]
            session_id="test",
            latency_budget=timedelta(milliseconds=100),
            traces=[],
            slow_traces=[],
        )
        as_dict = attrs.asdict(report)
        assert "session_id" in as_dict
        assert "latency_budget" in as_dict
        assert "traces" in as_dict
        assert "slow_traces" in as_dict


def _make_compute_sample(device: ComputeUnit = ComputeUnit.CPU) -> ComputeUnitUsageSample:
    return ComputeUnitUsageSample(
        timestamp=datetime.now(tz=UTC),
        device=device,
        usage_pct=10.0,
        frequency_mhz=1000.0,
        memory_mb=512.0,
        power_mw=5.0,
    )


def _make_memory_sample() -> MemoryUsageSample:
    return MemoryUsageSample(
        rss_bytes=1,
        vms_bytes=2,
        shared_bytes=3,
        text_bytes=4,
        lib_bytes=5,
        data_bytes=6,
        dirty_bytes=7,
    )


@pytest.mark.unit
class TestMemoryUsageSampleJson:
    """Tests for MemoryUsageSample.json()."""

    def test_json_returns_all_fields(self) -> None:
        """MemoryUsageSample.json returns all numeric fields."""
        sample = _make_memory_sample()
        result = sample.json()
        assert result == {
            "rss_bytes": 1,
            "vms_bytes": 2,
            "shared_bytes": 3,
            "text_bytes": 4,
            "lib_bytes": 5,
            "data_bytes": 6,
            "dirty_bytes": 7,
        }


@pytest.mark.unit
class TestResourceUsageSampleJson:
    """Tests for ResourceUsageSample.json()."""

    def test_json_returns_all_fields(self) -> None:
        """ResourceUsageSample.json returns nested json for sub-objects."""
        ts = datetime.now(tz=UTC)
        sample = ResourceUsageSample(
            timestamp=ts,
            running_span_id=42,
            cpu_usage=_make_compute_sample(ComputeUnit.CPU),
            gpu_usage=_make_compute_sample(ComputeUnit.GPU),
            npu_usage=_make_compute_sample(ComputeUnit.NPU),
            dsp_usage=_make_compute_sample(ComputeUnit.DSP),
            proc_cpu_usage=3.5,
            mem_usage=_make_memory_sample(),
        )
        result = sample.json()
        assert result["timestamp"] == ts.isoformat()
        assert result["running_span_id"] == 42
        assert result["proc_cpu_usage"] == 3.5
        assert result["cpu_usage"]["device"] == ComputeUnit.CPU
        assert result["gpu_usage"]["device"] == ComputeUnit.GPU
        assert result["npu_usage"]["device"] == ComputeUnit.NPU
        assert result["dsp_usage"]["device"] == ComputeUnit.DSP
        assert result["mem_usage"]["rss_bytes"] == 1


@pytest.mark.unit
class TestTraceSummaryWithResourceSamples:
    """Tests for Trace summary methods when resource_usage_samples is populated."""

    def _make_trace_with_samples(self, utc_past: datetime, utc_now: datetime) -> Trace:
        sample = ResourceUsageSample(
            timestamp=utc_now,
            running_span_id=None,
            cpu_usage=_make_compute_sample(),
            gpu_usage=_make_compute_sample(ComputeUnit.GPU),
            npu_usage=_make_compute_sample(ComputeUnit.NPU),
            dsp_usage=_make_compute_sample(ComputeUnit.DSP),
            proc_cpu_usage=0.0,
            mem_usage=_make_memory_sample(),
        )
        return Trace(
            id_=1,
            start=utc_past,
            end=utc_now,
            latency_ns=100_000_000,
            resource_usage_samples=[sample],
        )

    def test_summary_includes_resource_sample_count(
        self, utc_past: datetime, utc_now: datetime
    ) -> None:
        """Trace.summary includes the aggregated resource usage block."""
        trace = self._make_trace_with_samples(utc_past, utc_now)
        result = trace.summary()
        assert "Resource usage (1 samples)" in result
        assert "CPU:" in result
        assert "GPU:" in result
        assert "NPU:" in result
        assert "DSP:" in result
        assert "proc CPU:" in result
        assert "RSS:" in result

    def test_summary_rich_includes_resource_sample_count(
        self, utc_past: datetime, utc_now: datetime
    ) -> None:
        """Trace.summary_rich includes the aggregated resource usage block with rich formatting."""
        trace = self._make_trace_with_samples(utc_past, utc_now)
        result = trace.summary_rich()
        assert "Resource usage" in result
        assert "1 samples" in result
        assert "[bold]" in result
