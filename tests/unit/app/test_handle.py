"""Unit tests for PipelineHandle."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from moment_to_action.app._handle import PipelineHandle
from moment_to_action.hardware import ComputeUnit
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.metrics import MetricsCollector
from moment_to_action.pipeline import Pipeline
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moment_to_action.messages import Message


class _PassthroughStage(Stage):
    """A concrete stage that passes each message through unchanged."""

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Yield each buffered item unchanged."""
        yield from items


def _msg(timestamp: float = 0.0) -> RawFrameMessage:
    """Build a minimal RawFrameMessage for testing."""
    return RawFrameMessage(frame=None, timestamp=timestamp)


@pytest.mark.unit
class TestPipelineHandle:
    """Tests for PipelineHandle."""

    def test_starts_unloaded(self) -> None:
        """A freshly constructed handle is not loaded."""
        metrics = MetricsCollector(session_id="handle_test")
        stage = _PassthroughStage(metrics=metrics)
        handle = PipelineHandle(
            name="p", pipeline=Pipeline([stage], metrics=metrics), metrics=metrics, stage_units=[]
        )
        assert handle.loaded is False

    def test_load_calls_stage_load_and_sets_loaded(self) -> None:
        """load() calls stage.load(platform, unit) for every stage and flips loaded."""
        metrics = MetricsCollector(session_id="handle_test")
        stage = MagicMock(spec=Stage)
        platform = MagicMock()
        handle = PipelineHandle(
            name="p",
            pipeline=Pipeline([], metrics=metrics),
            metrics=metrics,
            stage_units=[(stage, ComputeUnit.CPU)],
        )

        handle.load(platform)

        stage.load.assert_called_once_with(platform, ComputeUnit.CPU)
        assert handle.loaded is True

    def test_load_twice_raises(self) -> None:
        """Loading an already-loaded pipeline raises RuntimeError."""
        metrics = MetricsCollector(session_id="handle_test")
        handle = PipelineHandle(
            name="p", pipeline=Pipeline([], metrics=metrics), metrics=metrics, stage_units=[]
        )
        handle.load(MagicMock())
        with pytest.raises(RuntimeError, match="already loaded"):
            handle.load(MagicMock())

    def test_unload_calls_stage_unload_and_clears_loaded(self) -> None:
        """unload() calls stage.unload() for every stage and flips loaded back."""
        metrics = MetricsCollector(session_id="handle_test")
        stage = MagicMock(spec=Stage)
        handle = PipelineHandle(
            name="p",
            pipeline=Pipeline([], metrics=metrics),
            metrics=metrics,
            stage_units=[(stage, None)],
        )
        handle.load(MagicMock())

        handle.unload()

        stage.unload.assert_called_once()
        assert handle.loaded is False

    def test_run_before_load_raises(self) -> None:
        """run() on an unloaded pipeline raises RuntimeError."""
        metrics = MetricsCollector(session_id="handle_test")
        handle = PipelineHandle(
            name="p", pipeline=Pipeline([], metrics=metrics), metrics=metrics, stage_units=[]
        )
        with pytest.raises(RuntimeError, match="not loaded"):
            list(handle.run(iter([])))

    def test_run_after_load_yields_pipeline_output(self) -> None:
        """run() delegates to the wrapped Pipeline once loaded."""
        metrics = MetricsCollector(session_id="handle_test")
        stage = _PassthroughStage(metrics=metrics)
        handle = PipelineHandle(
            name="p", pipeline=Pipeline([stage], metrics=metrics), metrics=metrics, stage_units=[]
        )
        handle.load(MagicMock())

        msg = _msg()
        results = list(handle.run(iter([msg])))

        assert results == [msg]

    def test_metrics_report_returns_collector_report(self) -> None:
        """metrics_report() forwards to the pipeline's own collector."""
        metrics = MetricsCollector(session_id="handle_test")
        handle = PipelineHandle(
            name="p", pipeline=Pipeline([], metrics=metrics), metrics=metrics, stage_units=[]
        )
        report = handle.metrics_report()
        assert report.session_id == "handle_test"
