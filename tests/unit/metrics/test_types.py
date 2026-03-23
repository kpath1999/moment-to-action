"""Unit tests for metrics types."""

from __future__ import annotations

import attrs
import pytest

from moment_to_action.metrics._types import (
    CollectorReport,
    EventRecord,
    EventType,
    LatencyBudget,
    PipelineRecord,
    PipelineStats,
    StageRecord,
    StageStats,
)


@pytest.mark.unit
class TestEventType:
    """Tests for EventType enum."""

    def test_eventtype_trigger_fired(self) -> None:
        """Test EventType.TRIGGER_FIRED member."""
        assert hasattr(EventType, "TRIGGER_FIRED")
        assert isinstance(EventType.TRIGGER_FIRED, EventType)

    def test_eventtype_detection(self) -> None:
        """Test EventType.DETECTION member."""
        assert hasattr(EventType, "DETECTION")
        assert isinstance(EventType.DETECTION, EventType)

    def test_eventtype_false_positive(self) -> None:
        """Test EventType.FALSE_POSITIVE member."""
        assert hasattr(EventType, "FALSE_POSITIVE")
        assert isinstance(EventType.FALSE_POSITIVE, EventType)

    def test_eventtype_all_members(self) -> None:
        """Test that all EventType members are present."""
        members = [member.name for member in EventType]
        assert "TRIGGER_FIRED" in members
        assert "DETECTION" in members
        assert "FALSE_POSITIVE" in members
        assert len(members) == 3


@pytest.mark.unit
class TestEventRecord:
    """Tests for EventRecord attrs class."""

    def test_eventrecord_construction_basic(self) -> None:
        """Test basic EventRecord construction."""
        record = EventRecord(
            timestamp=1234567890.0,
            event_type="model_load",
            data={"model": "yolo_v8"},
        )
        assert record.timestamp == 1234567890.0
        assert record.event_type == "model_load"
        assert record.data == {"model": "yolo_v8"}

    def test_eventrecord_construction_empty_data(self) -> None:
        """Test EventRecord construction with default empty data."""
        record = EventRecord(
            timestamp=1234567890.0,
            event_type="test_event",
        )
        assert record.data == {}

    def test_eventrecord_construction_complex_data(self) -> None:
        """Test EventRecord construction with complex data."""
        data = {
            "version": "1.0",
            "duration": 42.5,
            "success": True,
            "nested": {"key": "value"},
        }
        record = EventRecord(
            timestamp=1000.0,
            event_type="pipeline_event",
            data=data,
        )
        assert record.data == data
        assert record.data["nested"]["key"] == "value"

    def test_eventrecord_is_attrs_class(self) -> None:
        """Test that EventRecord is an attrs class."""
        assert attrs.has(EventRecord)

    def test_eventrecord_field_access(self) -> None:
        """Test field access on EventRecord."""
        record = EventRecord(
            timestamp=999.99,
            event_type="test",
        )
        assert hasattr(record, "timestamp")
        assert hasattr(record, "event_type")
        assert hasattr(record, "data")

    def test_eventrecord_asdict(self) -> None:
        """Test attrs.asdict serialization of EventRecord."""
        record = EventRecord(
            timestamp=500.0,
            event_type="checkpoint",
            data={"step": 100},
        )
        as_dict = attrs.asdict(record)
        assert as_dict["timestamp"] == 500.0
        assert as_dict["event_type"] == "checkpoint"
        assert as_dict["data"] == {"step": 100}


@pytest.mark.unit
class TestPipelineRecord:
    """Tests for PipelineRecord attrs class."""

    def test_pipelinerecord_construction_basic(self) -> None:
        """Test basic PipelineRecord construction."""
        record = PipelineRecord(
            timestamp=1234567890.0,
            event_type=EventType.DETECTION,
            latency_ms=42.5,
            metadata={"class": "person"},
        )
        assert record.timestamp == 1234567890.0
        assert record.event_type == EventType.DETECTION
        assert record.latency_ms == 42.5
        assert record.metadata == {"class": "person"}

    def test_pipelinerecord_construction_default_metadata(self) -> None:
        """Test PipelineRecord with default empty metadata."""
        record = PipelineRecord(
            timestamp=1000.0,
            event_type=EventType.TRIGGER_FIRED,
            latency_ms=100.0,
        )
        assert record.metadata == {}

    def test_pipelinerecord_is_attrs_class(self) -> None:
        """Test that PipelineRecord is an attrs class."""
        assert attrs.has(PipelineRecord)

    def test_pipelinerecord_asdict(self) -> None:
        """Test attrs.asdict serialization of PipelineRecord."""
        record = PipelineRecord(
            timestamp=2000.0,
            event_type=EventType.FALSE_POSITIVE,
            latency_ms=50.0,
            metadata={"confidence": 0.95},
        )
        as_dict = attrs.asdict(record)
        assert as_dict["timestamp"] == 2000.0
        assert as_dict["latency_ms"] == 50.0
        assert as_dict["metadata"]["confidence"] == 0.95


@pytest.mark.unit
class TestStageRecord:
    """Tests for StageRecord attrs class."""

    def test_stagerecord_construction_basic(self) -> None:
        """Test basic StageRecord construction."""
        record = StageRecord(
            timestamp=1234567890.0,
            stage_name="YoloDetector",
            stage_idx=0,
            latency_ms=25.5,
            metadata={"model": "yolo_v8"},
        )
        assert record.timestamp == 1234567890.0
        assert record.stage_name == "YoloDetector"
        assert record.stage_idx == 0
        assert record.latency_ms == 25.5
        assert record.metadata == {"model": "yolo_v8"}

    def test_stagerecord_construction_default_metadata(self) -> None:
        """Test StageRecord with default empty metadata."""
        record = StageRecord(
            timestamp=3000.0,
            stage_name="PreprocessStage",
            stage_idx=1,
            latency_ms=15.0,
        )
        assert record.metadata == {}

    def test_stagerecord_is_attrs_class(self) -> None:
        """Test that StageRecord is an attrs class."""
        assert attrs.has(StageRecord)

    def test_stagerecord_asdict(self) -> None:
        """Test attrs.asdict serialization of StageRecord."""
        record = StageRecord(
            timestamp=4000.0,
            stage_name="ReasoningStage",
            stage_idx=2,
            latency_ms=100.5,
            metadata={"tokens": 256},
        )
        as_dict = attrs.asdict(record)
        assert as_dict["stage_name"] == "ReasoningStage"
        assert as_dict["stage_idx"] == 2
        assert as_dict["metadata"]["tokens"] == 256


@pytest.mark.unit
class TestStageStats:
    """Tests for StageStats attrs class."""

    def test_stagestats_construction(self) -> None:
        """Test StageStats construction."""
        stats = StageStats(
            num_calls=100,
            mean_ms=25.5,
            p50_ms=20.0,
            p95_ms=45.0,
            min_ms=10.0,
            max_ms=100.0,
        )
        assert stats.num_calls == 100
        assert stats.mean_ms == 25.5
        assert stats.p50_ms == 20.0
        assert stats.p95_ms == 45.0
        assert stats.min_ms == 10.0
        assert stats.max_ms == 100.0

    def test_stagestats_is_attrs_class(self) -> None:
        """Test that StageStats is an attrs class."""
        assert attrs.has(StageStats)

    def test_stagestats_asdict(self) -> None:
        """Test attrs.asdict serialization of StageStats."""
        stats = StageStats(
            num_calls=50,
            mean_ms=30.0,
            p50_ms=25.0,
            p95_ms=50.0,
            min_ms=15.0,
            max_ms=120.0,
        )
        as_dict = attrs.asdict(stats)
        assert as_dict["num_calls"] == 50
        assert as_dict["mean_ms"] == 30.0
        assert as_dict["p95_ms"] == 50.0


@pytest.mark.unit
class TestPipelineStats:
    """Tests for PipelineStats attrs class."""

    def test_pipelinestats_construction(self) -> None:
        """Test PipelineStats construction."""
        stats = PipelineStats(
            total_triggers=100,
            total_detections=250,
            total_false_positives=5,
            trigger_rate=0.2,
            false_positive_rate=0.02,
        )
        assert stats.total_triggers == 100
        assert stats.total_detections == 250
        assert stats.total_false_positives == 5
        assert stats.trigger_rate == 0.2
        assert stats.false_positive_rate == 0.02

    def test_pipelinestats_is_attrs_class(self) -> None:
        """Test that PipelineStats is an attrs class."""
        assert attrs.has(PipelineStats)

    def test_pipelinestats_asdict(self) -> None:
        """Test attrs.asdict serialization of PipelineStats."""
        stats = PipelineStats(
            total_triggers=50,
            total_detections=200,
            total_false_positives=3,
            trigger_rate=0.15,
            false_positive_rate=0.015,
        )
        as_dict = attrs.asdict(stats)
        assert as_dict["total_triggers"] == 50
        assert as_dict["total_detections"] == 200
        assert as_dict["trigger_rate"] == 0.15


@pytest.mark.unit
class TestLatencyBudget:
    """Tests for LatencyBudget attrs class."""

    def test_latencybudget_construction_within_budget(self) -> None:
        """Test LatencyBudget construction when within budget."""
        budget = LatencyBudget(
            total_mean_ms=25.0,
            budget_ms=50.0,
            headroom_ms=25.0,
            within_budget=True,
        )
        assert budget.total_mean_ms == 25.0
        assert budget.budget_ms == 50.0
        assert budget.headroom_ms == 25.0
        assert budget.within_budget is True

    def test_latencybudget_construction_exceeds_budget(self) -> None:
        """Test LatencyBudget construction when exceeding budget."""
        budget = LatencyBudget(
            total_mean_ms=75.0,
            budget_ms=50.0,
            headroom_ms=-25.0,
            within_budget=False,
        )
        assert budget.total_mean_ms == 75.0
        assert budget.within_budget is False

    def test_latencybudget_is_attrs_class(self) -> None:
        """Test that LatencyBudget is an attrs class."""
        assert attrs.has(LatencyBudget)

    def test_latencybudget_asdict(self) -> None:
        """Test attrs.asdict serialization of LatencyBudget."""
        budget = LatencyBudget(
            total_mean_ms=30.5,
            budget_ms=100.0,
            headroom_ms=69.5,
            within_budget=True,
        )
        as_dict = attrs.asdict(budget)
        assert as_dict["total_mean_ms"] == 30.5
        assert as_dict["budget_ms"] == 100.0
        assert as_dict["within_budget"] is True


@pytest.mark.unit
class TestCollectorReport:
    """Tests for CollectorReport attrs class."""

    def test_collectorreport_construction(self) -> None:
        """Test CollectorReport construction with all fields."""
        per_stage = {
            "YoloDetector": StageStats(
                num_calls=100,
                mean_ms=25.0,
                p50_ms=20.0,
                p95_ms=45.0,
                min_ms=10.0,
                max_ms=100.0,
            ),
            "Preprocessor": StageStats(
                num_calls=100,
                mean_ms=15.0,
                p50_ms=12.0,
                p95_ms=25.0,
                min_ms=8.0,
                max_ms=50.0,
            ),
        }
        pipeline_stats = PipelineStats(
            total_triggers=50,
            total_detections=200,
            total_false_positives=2,
            trigger_rate=0.1,
            false_positive_rate=0.01,
        )
        latency_budget = LatencyBudget(
            total_mean_ms=40.0,
            budget_ms=100.0,
            headroom_ms=60.0,
            within_budget=True,
        )
        report = CollectorReport(
            session_id="session_123",
            total_stages=200,
            total_pipeline_events=500,
            per_stage=per_stage,
            pipeline=pipeline_stats,
            latency_budget=latency_budget,
        )
        assert report.session_id == "session_123"
        assert report.total_stages == 200
        assert report.total_pipeline_events == 500
        assert len(report.per_stage) == 2
        assert "YoloDetector" in report.per_stage
        assert report.pipeline.total_triggers == 50
        assert report.latency_budget.within_budget is True

    def test_collectorreport_is_attrs_class(self) -> None:
        """Test that CollectorReport is an attrs class."""
        assert attrs.has(CollectorReport)

    def test_collectorreport_asdict(self) -> None:
        """Test attrs.asdict serialization of CollectorReport."""
        per_stage = {
            "Stage1": StageStats(
                num_calls=50,
                mean_ms=20.0,
                p50_ms=18.0,
                p95_ms=35.0,
                min_ms=10.0,
                max_ms=80.0,
            )
        }
        pipeline_stats = PipelineStats(
            total_triggers=20,
            total_detections=100,
            total_false_positives=1,
            trigger_rate=0.05,
            false_positive_rate=0.01,
        )
        latency_budget = LatencyBudget(
            total_mean_ms=25.0,
            budget_ms=50.0,
            headroom_ms=25.0,
            within_budget=True,
        )
        report = CollectorReport(
            session_id="test_session",
            total_stages=100,
            total_pipeline_events=250,
            per_stage=per_stage,
            pipeline=pipeline_stats,
            latency_budget=latency_budget,
        )
        as_dict = attrs.asdict(report)
        assert as_dict["session_id"] == "test_session"
        assert as_dict["total_stages"] == 100
        assert "Stage1" in as_dict["per_stage"]

    def test_collectorreport_per_stage_access(self) -> None:
        """Test accessing per_stage statistics from CollectorReport."""
        stage_stats = StageStats(
            num_calls=75,
            mean_ms=22.5,
            p50_ms=20.0,
            p95_ms=40.0,
            min_ms=12.0,
            max_ms=90.0,
        )
        per_stage = {"CustomStage": stage_stats}
        pipeline_stats = PipelineStats(
            total_triggers=10,
            total_detections=50,
            total_false_positives=0,
            trigger_rate=0.02,
            false_positive_rate=0.0,
        )
        latency_budget = LatencyBudget(
            total_mean_ms=30.0,
            budget_ms=60.0,
            headroom_ms=30.0,
            within_budget=True,
        )
        report = CollectorReport(
            session_id="test",
            total_stages=150,
            total_pipeline_events=300,
            per_stage=per_stage,
            pipeline=pipeline_stats,
            latency_budget=latency_budget,
        )
        custom_stats = report.per_stage["CustomStage"]
        assert custom_stats.num_calls == 75
        assert custom_stats.mean_ms == 22.5


@pytest.mark.unit
class TestMetricsTypesConstructionDefaults:
    """Tests for construction with various default scenarios."""

    def test_eventrecord_with_defaults(self) -> None:
        """Test EventRecord with factory defaults."""
        record = EventRecord(
            timestamp=1234.0,
            event_type="test",
        )
        assert isinstance(record.data, dict)
        assert len(record.data) == 0

    def test_pipelinerecord_with_defaults(self) -> None:
        """Test PipelineRecord with factory defaults."""
        record = PipelineRecord(
            timestamp=1234.0,
            event_type=EventType.DETECTION,
            latency_ms=50.0,
        )
        assert isinstance(record.metadata, dict)
        assert len(record.metadata) == 0

    def test_stagerecord_with_defaults(self) -> None:
        """Test StageRecord with factory defaults."""
        record = StageRecord(
            timestamp=1234.0,
            stage_name="TestStage",
            stage_idx=0,
            latency_ms=25.0,
        )
        assert isinstance(record.metadata, dict)
        assert len(record.metadata) == 0

    def test_factory_dict_independence(self) -> None:
        """Test that factory dicts are independent across instances."""
        record1 = EventRecord(
            timestamp=100.0,
            event_type="event1",
        )
        record2 = EventRecord(
            timestamp=200.0,
            event_type="event2",
        )
        record1.data["key"] = "value1"
        assert "key" not in record2.data
        assert record2.data == {}
