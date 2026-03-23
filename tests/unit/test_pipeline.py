"""Unit tests for Pipeline class."""

from __future__ import annotations

from unittest import mock

import pytest

from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.pipeline import Pipeline


@pytest.mark.unit
class TestPipeline:
    """Tests for Pipeline class."""

    @pytest.fixture
    def sample_message(self) -> RawFrameMessage:
        """Create a sample RawFrameMessage for testing."""
        import time

        import numpy as np

        return RawFrameMessage(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            source="test_source",
            width=640,
            height=480,
            timestamp=time.time(),
        )

    def test_pipeline_run_with_mock_stages_in_order(self, sample_message: RawFrameMessage) -> None:
        """Test Pipeline.run() executes stages in order and passes messages through."""
        # Create three mock stages
        stage1 = mock.MagicMock()
        stage2 = mock.MagicMock()
        stage3 = mock.MagicMock()

        # Each stage returns a modified message to pass to the next
        result_msg1 = sample_message.model_copy(update={"latency_ms": 10.0})
        result_msg2 = sample_message.model_copy(update={"latency_ms": 20.0})
        result_msg3 = sample_message.model_copy(update={"latency_ms": 30.0})

        stage1.process.return_value = result_msg1
        stage2.process.return_value = result_msg2
        stage3.process.return_value = result_msg3

        pipeline = Pipeline([stage1, stage2, stage3])
        result = pipeline.run(sample_message)

        # Verify all stages were called
        stage1.process.assert_called_once()
        stage2.process.assert_called_once()
        stage3.process.assert_called_once()

        # Verify they were called in order with correct stage indices
        call_args1 = stage1.process.call_args
        call_args2 = stage2.process.call_args
        call_args3 = stage3.process.call_args

        assert call_args1.kwargs["stage_idx"] == 0
        assert call_args2.kwargs["stage_idx"] == 1
        assert call_args3.kwargs["stage_idx"] == 2

        # Verify final result is from last stage
        assert result == result_msg3
        assert result.latency_ms == 30.0

    def test_pipeline_none_short_circuit(self, sample_message: RawFrameMessage) -> None:
        """Test that None from a stage stops pipeline and returns None."""
        stage1 = mock.MagicMock()
        stage2 = mock.MagicMock()
        stage3 = mock.MagicMock()

        # Stage1 processes normally, Stage2 returns None (stops pipeline)
        result_msg1 = sample_message.model_copy(update={"latency_ms": 10.0})
        stage1.process.return_value = result_msg1
        stage2.process.return_value = None  # This stops the pipeline
        stage3.process.return_value = sample_message

        pipeline = Pipeline([stage1, stage2, stage3])
        result = pipeline.run(sample_message)

        # Verify result is None
        assert result is None

        # Verify Stage3 was never called (pipeline short-circuited)
        stage1.process.assert_called_once()
        stage2.process.assert_called_once()
        stage3.process.assert_not_called()

    def test_pipeline_metrics_forwarding(self, sample_message: RawFrameMessage) -> None:
        """Test that MetricsCollector is forwarded to all stages."""
        stage1 = mock.MagicMock()
        stage2 = mock.MagicMock()
        metrics_mock = mock.MagicMock()

        result_msg = sample_message.model_copy(update={"latency_ms": 15.0})
        stage1.process.return_value = result_msg
        stage2.process.return_value = result_msg

        pipeline = Pipeline([stage1, stage2], metrics=metrics_mock)
        pipeline.run(sample_message)

        # Verify metrics was passed to both stages
        call_args1 = stage1.process.call_args
        call_args2 = stage2.process.call_args

        assert call_args1.kwargs["metrics"] is metrics_mock
        assert call_args2.kwargs["metrics"] is metrics_mock

    def test_pipeline_empty_returns_input_unchanged(self, sample_message: RawFrameMessage) -> None:
        """Test that empty pipeline (no stages) returns input message unchanged."""
        pipeline = Pipeline([])
        result = pipeline.run(sample_message)

        # Result should be the same message object
        assert result == sample_message
        assert result is sample_message

    def test_pipeline_properties(self) -> None:
        """Test Pipeline properties (stages, metrics)."""
        stage1 = mock.MagicMock()
        stage2 = mock.MagicMock()
        metrics_mock = mock.MagicMock()

        pipeline = Pipeline([stage1, stage2], metrics=metrics_mock)

        assert pipeline.stages == [stage1, stage2]
        assert pipeline.metrics is metrics_mock

    def test_pipeline_none_metrics_is_optional(self, sample_message: RawFrameMessage) -> None:
        """Test that MetricsCollector is optional (None is passed to stages)."""
        stage1 = mock.MagicMock()
        result_msg = sample_message.model_copy(update={"latency_ms": 10.0})
        stage1.process.return_value = result_msg

        # Create pipeline with metrics=None
        pipeline = Pipeline([stage1], metrics=None)
        pipeline.run(sample_message)

        # Verify metrics=None was passed to stage
        call_args = stage1.process.call_args
        assert call_args.kwargs["metrics"] is None

    def test_pipeline_message_flow_through_stages(self, sample_message: RawFrameMessage) -> None:
        """Test that message flows correctly through multiple stages."""
        # Create stages that modify the latency to track flow
        stage1 = mock.MagicMock()
        stage2 = mock.MagicMock()

        msg_after_stage1 = sample_message.model_copy(update={"latency_ms": 5.0})
        msg_after_stage2 = sample_message.model_copy(update={"latency_ms": 10.0})

        stage1.process.return_value = msg_after_stage1
        stage2.process.return_value = msg_after_stage2

        pipeline = Pipeline([stage1, stage2])
        result = pipeline.run(sample_message)

        # Verify stage2 received the output from stage1
        stage2_input = stage2.process.call_args[0][0]
        assert stage2_input == msg_after_stage1

        # Verify final result is from stage2
        assert result == msg_after_stage2
