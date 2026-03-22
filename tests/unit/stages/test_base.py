"""Unit tests for Stage base class."""

from __future__ import annotations

import time
from unittest import mock

import numpy as np
import pytest

from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.stages._base import Stage


@pytest.mark.unit
class TestStage:
    """Tests for Stage base class."""

    @pytest.fixture
    def sample_message(self) -> RawFrameMessage:
        """Create a sample message for testing."""
        return RawFrameMessage(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            source="test",
            width=640,
            height=480,
            timestamp=time.time(),
        )

    @pytest.fixture
    def concrete_stage_class(self) -> type[Stage]:
        """Create a concrete Stage subclass for testing."""

        class ConcreteStage(Stage):
            """A concrete stage that passes through the message unchanged."""

            def _process(self, msg: RawFrameMessage) -> RawFrameMessage:
                # Simulate some minimal work
                time.sleep(0.001)
                return msg

        return ConcreteStage

    def test_stage_name_property_returns_class_name(
        self, concrete_stage_class: type[Stage]
    ) -> None:
        """Test that Stage.name property returns the class name."""
        stage = concrete_stage_class()
        assert stage.name == "ConcreteStage"

    def test_stage_process_stamps_latency_on_result(
        self, sample_message: RawFrameMessage, concrete_stage_class: type[Stage]
    ) -> None:
        """Test that Stage.process() stamps latency_ms on the result."""
        stage = concrete_stage_class()

        # Verify initial message has latency_ms = 0.0
        assert sample_message.latency_ms == 0.0

        # Process the message
        result = stage.process(sample_message, stage_idx=0)

        # Verify result is not None
        assert result is not None

        # Verify latency_ms was stamped and is > 0
        assert result.latency_ms > 0.0

        # Verify the latency is reasonable (> 1ms due to sleep)
        assert result.latency_ms >= 1.0

    def test_stage_process_logs_to_metrics_collector(
        self, sample_message: RawFrameMessage, concrete_stage_class: type[Stage]
    ) -> None:
        """Test that Stage.process() logs to MetricsCollector if provided."""
        stage = concrete_stage_class()
        metrics_mock = mock.MagicMock()

        stage.process(sample_message, stage_idx=1, metrics=metrics_mock)

        # Verify metrics.log_stage was called
        metrics_mock.log_stage.assert_called_once()

        # Verify call arguments
        call_args = metrics_mock.log_stage.call_args
        stage_name, stage_idx, latency_ms = call_args[0]

        assert stage_name == "ConcreteStage"
        assert stage_idx == 1
        assert latency_ms > 0.0

    def test_stage_process_returns_none_propagation(self) -> None:
        """Test that Stage.process() returns None if _process returns None."""

        class NoneStage(Stage):
            """A stage that returns None."""

            def _process(self, msg: RawFrameMessage) -> None:  # noqa: ARG002
                return None

        import time

        message = RawFrameMessage(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            source="test",
            width=640,
            height=480,
            timestamp=time.time(),
        )

        stage = NoneStage()
        result = stage.process(message)

        # Verify result is None
        assert result is None

    def test_stage_process_no_metrics_optional(
        self, sample_message: RawFrameMessage, concrete_stage_class: type[Stage]
    ) -> None:
        """Test that Stage.process() works without MetricsCollector."""
        stage = concrete_stage_class()

        # Process with metrics=None (should not raise)
        result = stage.process(sample_message, stage_idx=0, metrics=None)

        # Verify latency was still stamped
        assert result is not None
        assert result.latency_ms > 0.0

    def test_stage_process_stage_index_passed_correctly(
        self, sample_message: RawFrameMessage
    ) -> None:
        """Test that stage_idx is passed correctly to metrics."""

        class MockedStage(Stage):
            def _process(self, msg: RawFrameMessage) -> RawFrameMessage:
                return msg

        stage = MockedStage()
        metrics_mock = mock.MagicMock()

        # Test with different stage indices
        for idx in [0, 1, 5, 10]:
            stage.process(sample_message, stage_idx=idx, metrics=metrics_mock)
            # Get the most recent call
            call_args = metrics_mock.log_stage.call_args
            assert call_args[0][1] == idx  # Second positional arg is stage_idx

    def test_stage_process_preserves_message_data(self, concrete_stage_class: type[Stage]) -> None:
        """Test that Stage.process() preserves message data except latency_ms."""
        original_message = RawFrameMessage(
            frame=np.ones((480, 640, 3), dtype=np.uint8) * 100,
            source="test_source",
            width=640,
            height=480,
            timestamp=12345.0,
        )

        stage = concrete_stage_class()
        result = stage.process(original_message)

        # Verify data is preserved
        assert result.source == "test_source"
        assert result.width == 640
        assert result.height == 480
        assert result.timestamp == 12345.0
        assert np.array_equal(result.frame, original_message.frame)

    def test_stage_process_timing_accuracy(self, sample_message: RawFrameMessage) -> None:
        """Test that Stage.process() measures latency accurately."""

        class SlowStage(Stage):
            def _process(self, msg: RawFrameMessage) -> RawFrameMessage:
                time.sleep(0.05)  # 50ms
                return msg

        stage = SlowStage()
        result = stage.process(sample_message)

        # Verify latency is approximately 50ms (allow 10ms margin)
        assert 40.0 <= result.latency_ms <= 100.0

    def test_stage_process_returns_same_message_type(
        self, concrete_stage_class: type[Stage]
    ) -> None:
        """Test that Stage.process() returns the same message type."""
        import time

        message = RawFrameMessage(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            source="test",
            width=640,
            height=480,
            timestamp=time.time(),
        )

        stage = concrete_stage_class()
        result = stage.process(message)

        # Verify type is preserved
        assert isinstance(result, RawFrameMessage)
