"""Unit tests for BasePreprocessor class."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from moment_to_action.hardware import ComputeUnit
from moment_to_action.stages._preprocess import BasePreprocessor
from moment_to_action.utils import BufferPool, BufferSpec


@pytest.mark.unit
class TestBasePreprocessor:
    """Tests for BasePreprocessor class."""

    @pytest.fixture
    def concrete_preprocessor_class(
        self,
    ) -> type[BasePreprocessor[np.ndarray, np.ndarray]]:
        """Create a concrete BasePreprocessor subclass for testing."""

        class TestPreprocessor(BasePreprocessor[np.ndarray, np.ndarray]):
            """Test preprocessor that simply doubles input."""

            def _validate(self, data: np.ndarray) -> None:
                """Validate that input is a numpy array."""
                if not isinstance(data, np.ndarray):
                    msg = "Input must be a numpy array"
                    raise TypeError(msg)

            def _allocate_buffers(self) -> None:
                """Allocate a simple output buffer."""
                self._buffers.register("output", BufferSpec(shape=(10,), dtype=np.float32))

            def _process(self, data: np.ndarray) -> np.ndarray:
                """Double the input."""
                return data * 2

        return TestPreprocessor

    def test_allocate_buffers_called_on_construction(
        self, concrete_preprocessor_class: type[BasePreprocessor[np.ndarray, np.ndarray]]
    ) -> None:
        """Test that _allocate_buffers() is called on __init__."""
        # The fact that this doesn't raise means _allocate_buffers was called
        preprocessor = concrete_preprocessor_class()

        # Verify buffers were registered
        assert preprocessor.buffer_pool.total_bytes > 0

        # Verify we can get the buffer
        buf = preprocessor.buffer_pool.get("output")
        assert buf is not None
        assert buf.shape == (10,)
        assert buf.dtype == np.float32

    def test_preprocessor_caches_buffers_across_calls(
        self, concrete_preprocessor_class: type[BasePreprocessor[np.ndarray, np.ndarray]]
    ) -> None:
        """Test that preprocessor caches buffers and reuses them across calls."""
        preprocessor = concrete_preprocessor_class()

        # Get buffer reference
        buf1 = preprocessor.buffer_pool.get("output")
        buf1_id = id(buf1)

        # Call process twice
        input_data = np.array([1.0, 2.0, 3.0])
        preprocessor.process(input_data)
        preprocessor.process(input_data)

        # Get buffer again and verify it's the same object
        buf2 = preprocessor.buffer_pool.get("output")
        buf2_id = id(buf2)

        # Same buffer should be reused (same id)
        assert buf1_id == buf2_id

    def test_dispatch_routes_to_cpu_no_dsp_on_laptop(self) -> None:
        """Test that _dispatch() routes to CPU (no DSP on laptop)."""

        class DispatchTestPreprocessor(BasePreprocessor[np.ndarray, np.ndarray]):
            """Preprocessor that tests dispatch routing."""

            def _validate(self, data: np.ndarray) -> None:
                if not isinstance(data, np.ndarray):
                    msg = "Input must be a numpy array"
                    raise TypeError(msg)

            def _allocate_buffers(self) -> None:
                pass

            def _process(self, data: np.ndarray) -> np.ndarray:
                """Use _dispatch to call a function."""

                def multiply(x: np.ndarray) -> np.ndarray:
                    return x * 3

                # Should route to CPU since no DSP backend
                return self._dispatch(multiply, data)

        preprocessor = DispatchTestPreprocessor()

        # Verify compute unit is CPU
        assert preprocessor.compute_unit == ComputeUnit.CPU

        # Test that _dispatch works and routes to CPU
        input_data = np.array([2.0, 4.0])
        result = preprocessor.process(input_data)

        # Verify result is correct (multiply by 3)
        np.testing.assert_array_equal(result, np.array([6.0, 12.0]))

    def test_preprocessor_process_validates_input(
        self, concrete_preprocessor_class: type[BasePreprocessor[np.ndarray, np.ndarray]]
    ) -> None:
        """Test that preprocessor.process() validates input."""
        preprocessor = concrete_preprocessor_class()

        # Pass invalid input (not an array)
        with pytest.raises(TypeError, match="Input must be a numpy array"):
            preprocessor.process("not an array")

    def test_preprocessor_process_returns_correct_output(
        self, concrete_preprocessor_class: type[BasePreprocessor[np.ndarray, np.ndarray]]
    ) -> None:
        """Test that preprocessor.process() returns correct output."""
        preprocessor = concrete_preprocessor_class()

        input_data = np.array([1.0, 2.0, 3.0])
        result = preprocessor.process(input_data)

        # Verify result is doubled
        expected = input_data * 2
        np.testing.assert_array_equal(result, expected)

    def test_preprocessor_logs_to_metrics_collector(
        self, concrete_preprocessor_class: type[BasePreprocessor[np.ndarray, np.ndarray]]
    ) -> None:
        """Test that preprocessor.process() logs to MetricsCollector if provided."""
        metrics_mock = mock.MagicMock()
        preprocessor = concrete_preprocessor_class(metrics=metrics_mock)

        input_data = np.array([1.0, 2.0, 3.0])
        preprocessor.process(input_data)

        # Verify metrics.log_event was called
        metrics_mock.log_event.assert_called_once()

        # Verify call arguments
        call_args = metrics_mock.log_event.call_args
        event_type, data = call_args[0]

        assert event_type == "preprocess"
        assert "preprocessor" in data
        assert data["preprocessor"] == "TestPreprocessor"
        assert "latency_ms" in data
        assert "compute_unit" in data
        assert data["compute_unit"] == "CPU"

    def test_preprocessor_compute_unit_property(
        self, concrete_preprocessor_class: type[BasePreprocessor[np.ndarray, np.ndarray]]
    ) -> None:
        """Test that compute_unit property returns correct value."""
        preprocessor = concrete_preprocessor_class(compute_unit=ComputeUnit.CPU)
        assert preprocessor.compute_unit == ComputeUnit.CPU

    def test_preprocessor_buffer_pool_property(
        self, concrete_preprocessor_class: type[BasePreprocessor[np.ndarray, np.ndarray]]
    ) -> None:
        """Test that buffer_pool property returns the BufferPool."""
        preprocessor = concrete_preprocessor_class()
        pool = preprocessor.buffer_pool

        assert isinstance(pool, BufferPool)
        assert pool.total_bytes > 0

    def test_preprocessor_multiple_buffer_registration(self) -> None:
        """Test that preprocessor can register multiple buffers."""

        class MultiBufferPreprocessor(BasePreprocessor[np.ndarray, dict]):
            def _validate(self, data: np.ndarray) -> None:
                if not isinstance(data, np.ndarray):
                    msg = "Input must be a numpy array"
                    raise TypeError(msg)

            def _allocate_buffers(self) -> None:
                self._buffers.register("buffer1", BufferSpec(shape=(5,), dtype=np.float32))
                self._buffers.register("buffer2", BufferSpec(shape=(10,), dtype=np.int32))
                self._buffers.register("buffer3", BufferSpec(shape=(3, 3), dtype=np.uint8))

            def _process(self, data: np.ndarray) -> dict:  # noqa: ARG002
                # Just return a dict with buffer info
                return {"status": "ok"}

        preprocessor = MultiBufferPreprocessor()

        # Verify all buffers are accessible
        buf1 = preprocessor.buffer_pool.get("buffer1")
        buf2 = preprocessor.buffer_pool.get("buffer2")
        buf3 = preprocessor.buffer_pool.get("buffer3")

        assert buf1.shape == (5,)
        assert buf2.shape == (10,)
        assert buf3.shape == (3, 3)
        assert buf1.dtype == np.float32
        assert buf2.dtype == np.int32
        assert buf3.dtype == np.uint8

    def test_preprocessor_no_metrics_optional(
        self, concrete_preprocessor_class: type[BasePreprocessor[np.ndarray, np.ndarray]]
    ) -> None:
        """Test that MetricsCollector is optional."""
        # Create without metrics (should not raise)
        preprocessor = concrete_preprocessor_class(metrics=None)

        input_data = np.array([1.0, 2.0, 3.0])
        result = preprocessor.process(input_data)

        # Verify result is still correct
        np.testing.assert_array_equal(result, input_data * 2)

    def test_preprocessor_dispatch_with_kwargs(self) -> None:
        """Test that _dispatch correctly forwards kwargs."""

        class KwargsPreprocessor(BasePreprocessor[np.ndarray, np.ndarray]):
            def _validate(self, data: np.ndarray) -> None:
                if not isinstance(data, np.ndarray):
                    msg = "Input must be a numpy array"
                    raise TypeError(msg)

            def _allocate_buffers(self) -> None:
                pass

            def _process(self, data: np.ndarray) -> np.ndarray:
                """Test _dispatch with kwargs."""

                def add_and_multiply(x: np.ndarray, y: float, factor: int = 1) -> np.ndarray:
                    return (x + y) * factor

                # Dispatch with kwargs
                return self._dispatch(add_and_multiply, data, 10.0, factor=2)

        preprocessor = KwargsPreprocessor()
        input_data = np.array([1.0, 2.0, 3.0])
        result = preprocessor.process(input_data)

        # (input + 10) * 2 = (1+10)*2, (2+10)*2, (3+10)*2 = [22, 24, 26]
        expected = (input_data + 10.0) * 2
        np.testing.assert_array_equal(result, expected)
