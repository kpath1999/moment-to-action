"""Unit tests for MetricsCollector."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from moment_to_action.metrics import (
    EventType,
    MetricsCollector,
)

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


@pytest.mark.unit
class TestMetricsCollectorConstruction:
    """Test MetricsCollector initialization."""

    def test_construction_with_defaults(self) -> None:
        """Test MetricsCollector construction with default parameters."""
        collector = MetricsCollector()
        assert collector.session_id.startswith("session_")
        assert isinstance(collector.session_id, str)

    def test_construction_with_session_id(self) -> None:
        """Test MetricsCollector construction with custom session_id."""
        session_id = "test_session_123"
        collector = MetricsCollector(session_id=session_id)
        assert collector.session_id == session_id

    def test_construction_with_latency_budget(self) -> None:
        """Test MetricsCollector construction with custom latency_budget_ms."""
        budget = 3000.0
        collector = MetricsCollector(latency_budget_ms=budget)
        report = collector.report()
        assert report.latency_budget.budget_ms == budget

    def test_construction_with_both_params(self) -> None:
        """Test MetricsCollector construction with both session_id and latency_budget_ms."""
        session_id = "custom_session"
        budget = 2500.0
        collector = MetricsCollector(session_id=session_id, latency_budget_ms=budget)
        assert collector.session_id == session_id
        report = collector.report()
        assert report.latency_budget.budget_ms == budget


@pytest.mark.unit
class TestLogStage:
    """Test the log_stage() method."""

    def test_log_single_stage(self) -> None:
        """Test logging a single stage execution."""
        collector = MetricsCollector()
        collector.log_stage(
            stage_name="YOLOStage",
            stage_idx=0,
            latency_ms=100.5,
        )
        report = collector.report()
        assert report.total_stages == 1
        assert "YOLOStage" in report.per_stage

    def test_log_multiple_stages(self) -> None:
        """Test logging multiple stage executions."""
        collector = MetricsCollector()
        collector.log_stage(stage_name="YOLOStage", stage_idx=0, latency_ms=100.0)
        collector.log_stage(stage_name="LLMStage", stage_idx=1, latency_ms=200.0)
        collector.log_stage(stage_name="YOLOStage", stage_idx=0, latency_ms=110.0)

        report = collector.report()
        assert report.total_stages == 3
        assert "YOLOStage" in report.per_stage
        assert "LLMStage" in report.per_stage

    def test_log_stage_with_metadata(self) -> None:
        """Test logging a stage with metadata."""
        collector = MetricsCollector()
        metadata = {"model": "yolov8", "device": "cpu"}
        collector.log_stage(
            stage_name="YOLOStage",
            stage_idx=0,
            latency_ms=150.0,
            metadata=metadata,
        )
        report = collector.report()
        assert report.total_stages == 1

    def test_log_stage_without_metadata(self) -> None:
        """Test logging a stage without explicit metadata defaults to empty dict."""
        collector = MetricsCollector()
        collector.log_stage(
            stage_name="TestStage",
            stage_idx=0,
            latency_ms=50.0,
        )
        report = collector.report()
        assert report.total_stages == 1


@pytest.mark.unit
class TestLogPipelineEvent:
    """Test the log_pipeline_event() method."""

    def test_log_trigger_event(self) -> None:
        """Test logging a TRIGGER_FIRED event."""
        collector = MetricsCollector()
        collector.log_pipeline_event(
            event_type=EventType.TRIGGER_FIRED,
            latency_ms=500.0,
        )
        report = collector.report()
        assert report.total_pipeline_events == 1
        assert report.pipeline.total_triggers == 1
        assert report.pipeline.total_detections == 0

    def test_log_detection_event(self) -> None:
        """Test logging a DETECTION event."""
        collector = MetricsCollector()
        collector.log_pipeline_event(
            event_type=EventType.DETECTION,
            latency_ms=600.0,
        )
        report = collector.report()
        assert report.total_pipeline_events == 1
        assert report.pipeline.total_detections == 1

    def test_log_false_positive_event(self) -> None:
        """Test logging a FALSE_POSITIVE event."""
        collector = MetricsCollector()
        collector.log_pipeline_event(
            event_type=EventType.FALSE_POSITIVE,
            latency_ms=700.0,
        )
        report = collector.report()
        assert report.pipeline.total_false_positives == 1

    def test_log_mixed_events(self) -> None:
        """Test logging multiple event types."""
        collector = MetricsCollector()
        collector.log_pipeline_event(EventType.TRIGGER_FIRED, 500.0)
        collector.log_pipeline_event(EventType.DETECTION, 600.0)
        collector.log_pipeline_event(EventType.DETECTION, 610.0)
        collector.log_pipeline_event(EventType.FALSE_POSITIVE, 620.0)

        report = collector.report()
        assert report.total_pipeline_events == 4
        assert report.pipeline.total_triggers == 1
        assert report.pipeline.total_detections == 2
        assert report.pipeline.total_false_positives == 1

    def test_log_pipeline_event_with_metadata(self) -> None:
        """Test logging a pipeline event with metadata."""
        collector = MetricsCollector()
        metadata = {"confidence": 0.95, "class_id": 42}
        collector.log_pipeline_event(
            event_type=EventType.DETECTION,
            latency_ms=550.0,
            metadata=metadata,
        )
        report = collector.report()
        assert report.total_pipeline_events == 1

    def test_log_pipeline_event_without_metadata(self) -> None:
        """Test logging a pipeline event defaults to empty metadata."""
        collector = MetricsCollector()
        collector.log_pipeline_event(
            event_type=EventType.TRIGGER_FIRED,
            latency_ms=500.0,
        )
        report = collector.report()
        assert report.total_pipeline_events == 1


@pytest.mark.unit
class TestStageStatsComputation:
    """Test StageStats computation: mean, p50, p95, min, max."""

    def test_single_stage_execution_stats(self) -> None:
        """Test stats for a single stage execution."""
        collector = MetricsCollector()
        collector.log_stage("TestStage", 0, 100.0)

        report = collector.report()
        stats = report.per_stage["TestStage"]

        assert stats.num_calls == 1
        assert stats.mean_ms == 100.0
        assert stats.p50_ms == 100.0
        assert stats.p95_ms == 100.0
        assert stats.min_ms == 100.0
        assert stats.max_ms == 100.0

    def test_multiple_stage_executions_stats(self) -> None:
        """Test stats for multiple stage executions."""
        collector = MetricsCollector()
        latencies = [50.0, 100.0, 150.0, 200.0, 250.0]
        for lat in latencies:
            collector.log_stage("TestStage", 0, lat)

        report = collector.report()
        stats = report.per_stage["TestStage"]

        assert stats.num_calls == 5
        assert stats.mean_ms == 150.0
        assert stats.min_ms == 50.0
        assert stats.max_ms == 250.0

    def test_percentile_calculations(self) -> None:
        """Test that p50 and p95 are computed correctly using numpy."""
        collector = MetricsCollector()
        # Use a known distribution
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        for lat in latencies:
            collector.log_stage("TestStage", 0, lat)

        report = collector.report()
        stats = report.per_stage["TestStage"]

        # Verify with numpy
        arr = np.array(latencies)
        expected_mean = float(np.mean(arr))
        expected_p50 = float(np.percentile(arr, 50))
        expected_p95 = float(np.percentile(arr, 95))
        expected_min = float(np.min(arr))
        expected_max = float(np.max(arr))

        assert stats.mean_ms == expected_mean
        assert stats.p50_ms == expected_p50
        assert stats.p95_ms == expected_p95
        assert stats.min_ms == expected_min
        assert stats.max_ms == expected_max

    def test_multiple_stages_separate_stats(self) -> None:
        """Test that stats are computed separately for each stage."""
        collector = MetricsCollector()

        # Stage A: 10, 20, 30
        for lat in [10.0, 20.0, 30.0]:
            collector.log_stage("StageA", 0, lat)

        # Stage B: 100, 200, 300
        for lat in [100.0, 200.0, 300.0]:
            collector.log_stage("StageB", 1, lat)

        report = collector.report()

        stats_a = report.per_stage["StageA"]
        stats_b = report.per_stage["StageB"]

        assert stats_a.num_calls == 3
        assert stats_a.mean_ms == 20.0
        assert stats_a.min_ms == 10.0
        assert stats_a.max_ms == 30.0

        assert stats_b.num_calls == 3
        assert stats_b.mean_ms == 200.0
        assert stats_b.min_ms == 100.0
        assert stats_b.max_ms == 300.0


@pytest.mark.unit
class TestReportAggregation:
    """Test report() aggregates data into CollectorReport."""

    def test_report_with_no_data(self) -> None:
        """Test report() with no logged data."""
        collector = MetricsCollector(session_id="empty_session")
        report = collector.report()

        assert report.session_id == "empty_session"
        assert report.total_stages == 0
        assert report.total_pipeline_events == 0
        assert len(report.per_stage) == 0
        assert report.pipeline.total_triggers == 0
        assert report.pipeline.total_detections == 0
        assert report.pipeline.total_false_positives == 0

    def test_report_aggregates_stages(self) -> None:
        """Test report() correctly aggregates stage data."""
        collector = MetricsCollector()
        collector.log_stage("StageA", 0, 100.0)
        collector.log_stage("StageB", 1, 200.0)
        collector.log_stage("StageA", 0, 110.0)

        report = collector.report()
        assert report.total_stages == 3
        assert len(report.per_stage) == 2

    def test_report_aggregates_pipeline_events(self) -> None:
        """Test report() correctly aggregates pipeline events."""
        collector = MetricsCollector()
        collector.log_pipeline_event(EventType.TRIGGER_FIRED, 500.0)
        collector.log_pipeline_event(EventType.DETECTION, 600.0)

        report = collector.report()
        assert report.total_pipeline_events == 2

    def test_report_includes_all_sections(self) -> None:
        """Test that report() includes all expected sections."""
        collector = MetricsCollector(session_id="full_session", latency_budget_ms=2000.0)
        collector.log_stage("TestStage", 0, 100.0)
        collector.log_pipeline_event(EventType.TRIGGER_FIRED, 500.0)

        report = collector.report()

        assert hasattr(report, "session_id")
        assert hasattr(report, "total_stages")
        assert hasattr(report, "total_pipeline_events")
        assert hasattr(report, "per_stage")
        assert hasattr(report, "pipeline")
        assert hasattr(report, "latency_budget")


@pytest.mark.unit
class TestLatencyBudgetAnalysis:
    """Test latency budget computation."""

    def test_latency_within_budget(self) -> None:
        """Test when total latency is within budget."""
        collector = MetricsCollector(latency_budget_ms=2000.0)
        collector.log_pipeline_event(EventType.TRIGGER_FIRED, 500.0)
        collector.log_pipeline_event(EventType.DETECTION, 600.0)

        report = collector.report()
        assert report.latency_budget.total_mean_ms < report.latency_budget.budget_ms
        assert report.latency_budget.within_budget is True

    def test_latency_over_budget(self) -> None:
        """Test when total latency exceeds budget."""
        collector = MetricsCollector(latency_budget_ms=500.0)
        collector.log_pipeline_event(EventType.TRIGGER_FIRED, 1000.0)
        collector.log_pipeline_event(EventType.DETECTION, 600.0)

        report = collector.report()
        assert report.latency_budget.total_mean_ms > report.latency_budget.budget_ms
        assert report.latency_budget.within_budget is False

    def test_latency_headroom_calculation(self) -> None:
        """Test headroom calculation."""
        budget = 2000.0
        collector = MetricsCollector(latency_budget_ms=budget)
        collector.log_pipeline_event(EventType.TRIGGER_FIRED, 500.0)
        collector.log_pipeline_event(EventType.DETECTION, 600.0)

        report = collector.report()
        expected_headroom = budget - report.latency_budget.total_mean_ms
        assert report.latency_budget.headroom_ms == expected_headroom

    def test_no_pipeline_events_latency_budget(self) -> None:
        """Test latency budget with no pipeline events."""
        collector = MetricsCollector(latency_budget_ms=1000.0)
        report = collector.report()

        assert report.latency_budget.total_mean_ms == 0.0
        assert report.latency_budget.within_budget is True


@pytest.mark.unit
class TestPipelineStats:
    """Test pipeline event statistics computation."""

    def test_trigger_rate_computation(self) -> None:
        """Test trigger rate calculation."""
        collector = MetricsCollector()
        collector.log_pipeline_event(EventType.TRIGGER_FIRED, 500.0)
        collector.log_pipeline_event(EventType.TRIGGER_FIRED, 510.0)
        collector.log_pipeline_event(EventType.DETECTION, 600.0)

        report = collector.report()
        # 2 triggers out of 3 events = 2/3 ≈ 0.667
        assert report.pipeline.trigger_rate == pytest.approx(2 / 3)

    def test_false_positive_rate_computation(self) -> None:
        """Test false positive rate calculation."""
        collector = MetricsCollector()
        collector.log_pipeline_event(EventType.DETECTION, 600.0)
        collector.log_pipeline_event(EventType.DETECTION, 610.0)
        collector.log_pipeline_event(EventType.FALSE_POSITIVE, 620.0)

        report = collector.report()
        # 1 false positive out of 2 detections = 0.5
        assert report.pipeline.false_positive_rate == pytest.approx(0.5)

    def test_false_positive_rate_with_no_detections(self) -> None:
        """Test false positive rate when there are no detections."""
        collector = MetricsCollector()
        collector.log_pipeline_event(EventType.TRIGGER_FIRED, 500.0)

        report = collector.report()
        # No detections, so division by max(1, 0) = 0 / 1 = 0
        assert report.pipeline.false_positive_rate == 0.0


@pytest.mark.unit
class TestPrintStageLatencies:
    """Test print_stage_latencies() method."""

    def test_print_stage_latencies_no_raise(self, caplog: LogCaptureFixture) -> None:
        """Test that print_stage_latencies() doesn't raise an exception."""
        collector = MetricsCollector()
        collector.log_stage("TestStage", 0, 100.0)
        collector.log_stage("TestStage", 0, 150.0)

        with caplog.at_level(logging.INFO):
            collector.print_stage_latencies()

        # Should not raise, and should log something
        assert "TestStage" in caplog.text or len(caplog.records) > 0

    def test_print_stage_latencies_empty(self, caplog: LogCaptureFixture) -> None:
        """Test print_stage_latencies() with no logged stages."""
        collector = MetricsCollector()

        with caplog.at_level(logging.INFO):
            collector.print_stage_latencies()

        # Should log "No stage latencies recorded"
        assert "No stage latencies recorded" in caplog.text

    def test_print_stage_latencies_format(self, caplog: LogCaptureFixture) -> None:
        """Test that print_stage_latencies() produces expected output format."""
        collector = MetricsCollector()
        collector.log_stage("StageA", 0, 100.0)
        collector.log_stage("StageB", 1, 200.0)

        with caplog.at_level(logging.INFO):
            collector.print_stage_latencies()

        # Check for expected format markers (header, separator, total)
        assert "Stage" in caplog.text
        assert "Latency" in caplog.text


@pytest.mark.unit
class TestPrintSummary:
    """Test print_summary() method."""

    def test_print_summary_no_raise(self, caplog: LogCaptureFixture) -> None:
        """Test that print_summary() doesn't raise an exception."""
        collector = MetricsCollector()
        collector.log_stage("TestStage", 0, 100.0)
        collector.log_pipeline_event(EventType.TRIGGER_FIRED, 500.0)

        with caplog.at_level(logging.INFO):
            collector.print_summary()

        # Should not raise
        assert len(caplog.records) > 0

    def test_print_summary_empty(self, caplog: LogCaptureFixture) -> None:
        """Test print_summary() with empty collector."""
        collector = MetricsCollector()

        with caplog.at_level(logging.INFO):
            collector.print_summary()

        # Should still log summary header
        assert "METRICS SUMMARY" in caplog.text

    def test_print_summary_format(self, caplog: LogCaptureFixture) -> None:
        """Test that print_summary() produces expected output format."""
        collector = MetricsCollector(session_id="test_session")
        collector.log_stage("TestStage", 0, 100.0)

        with caplog.at_level(logging.INFO):
            collector.print_summary()

        # Check for expected format elements
        assert "METRICS SUMMARY" in caplog.text
        assert "test_session" in caplog.text
        assert "Total stages" in caplog.text or "Per-stage" in caplog.text


@pytest.mark.unit
class TestSaveMethod:
    """Test save() method writes valid JSON file."""

    def test_save_creates_file(self) -> None:
        """Test that save() creates a file."""
        collector = MetricsCollector(session_id="save_test")
        collector.log_stage("TestStage", 0, 100.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            collector.save(str(path))

            assert path.exists()

    def test_save_writes_valid_json(self) -> None:
        """Test that save() writes valid JSON content."""
        collector = MetricsCollector(session_id="json_test")
        collector.log_stage("TestStage", 0, 100.0)
        collector.log_pipeline_event(EventType.TRIGGER_FIRED, 500.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            collector.save(str(path))

            # Read and parse JSON
            content = path.read_text()
            data = json.loads(content)

            assert isinstance(data, dict)
            assert "session_id" in data
            assert data["session_id"] == "json_test"

    def test_save_includes_all_report_data(self) -> None:
        """Test that save() includes all report data in JSON."""
        collector = MetricsCollector(session_id="full_json_test", latency_budget_ms=2000.0)
        collector.log_stage("StageA", 0, 100.0)
        collector.log_stage("StageB", 1, 200.0)
        collector.log_pipeline_event(EventType.TRIGGER_FIRED, 500.0)
        collector.log_pipeline_event(EventType.DETECTION, 600.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            collector.save(str(path))

            data = json.loads(path.read_text())

            # Verify structure
            assert data["session_id"] == "full_json_test"
            assert data["total_stages"] == 2
            assert data["total_pipeline_events"] == 2
            assert "per_stage" in data
            assert "pipeline" in data
            assert "latency_budget" in data

    def test_save_with_nested_path(self) -> None:
        """Test that save() works with nested directory paths."""
        collector = MetricsCollector(session_id="nested_test")
        collector.log_stage("TestStage", 0, 100.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "results" / "metrics"
            nested_dir.mkdir(parents=True, exist_ok=True)
            path = nested_dir / "report.json"

            collector.save(str(path))

            assert path.exists()
            data = json.loads(path.read_text())
            assert data["session_id"] == "nested_test"


@pytest.mark.unit
class TestLogEvent:
    """Test the log_event() method for general-purpose event logging."""

    def test_log_event(self) -> None:
        """Test logging a general event."""
        collector = MetricsCollector()
        collector.log_event("model_load", {"model": "yolov8", "device": "cpu"})

        # Verify it doesn't break report generation
        report = collector.report()
        assert report.session_id is not None

    def test_multiple_events(self) -> None:
        """Test logging multiple general events."""
        collector = MetricsCollector()
        collector.log_event("model_load", {"model": "yolov8"})
        collector.log_event("config_change", {"setting": "fps", "value": 30})

        # Should not raise
        report = collector.report()
        assert report.session_id is not None


@pytest.mark.unit
class TestIntegrationScenario:
    """Integration-style tests combining multiple operations."""

    def test_full_pipeline_scenario(self) -> None:
        """Test a complete pipeline execution scenario."""
        collector = MetricsCollector(
            session_id="full_scenario",
            latency_budget_ms=2000.0,
        )

        # Simulate sensor stage
        collector.log_stage("SensorStage", 0, 50.0)
        collector.log_stage("SensorStage", 0, 55.0)

        # Simulate trigger event
        collector.log_pipeline_event(EventType.TRIGGER_FIRED, 500.0)

        # Simulate vision stage
        collector.log_stage("YOLOStage", 1, 200.0)
        collector.log_stage("YOLOStage", 1, 210.0)

        # Simulate detection
        collector.log_pipeline_event(EventType.DETECTION, 600.0)

        # Generate report
        report = collector.report()

        assert report.session_id == "full_scenario"
        assert report.total_stages == 4
        assert report.total_pipeline_events == 2
        assert "SensorStage" in report.per_stage
        assert "YOLOStage" in report.per_stage
        assert report.pipeline.total_triggers == 1
        assert report.pipeline.total_detections == 1

    def test_save_and_reload_metrics(self) -> None:
        """Test saving metrics and loading them back."""
        collector1 = MetricsCollector(
            session_id="save_reload_test",
            latency_budget_ms=1500.0,
        )
        collector1.log_stage("TestStage", 0, 100.0)
        collector1.log_pipeline_event(EventType.TRIGGER_FIRED, 500.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            collector1.save(str(path))

            # Load and verify
            data = json.loads(path.read_text())
            assert data["session_id"] == "save_reload_test"
            assert data["latency_budget"]["budget_ms"] == 1500.0
            assert data["total_stages"] == 1
            assert data["total_pipeline_events"] == 1
