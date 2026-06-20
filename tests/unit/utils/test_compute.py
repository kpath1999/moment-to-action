"""Tests for moment_to_action.utils.compute compute dispatcher."""

from __future__ import annotations

import logging

import pytest

from moment_to_action.hardware import ComputeUnit
from moment_to_action.utils.compute import ComputeDispatcher


@pytest.mark.unit
class TestComputeDispatcher:
    """Test ComputeDispatcher routing."""

    def test_cpu_path_default(self) -> None:
        """ComputeDispatcher should route to CPU by default."""
        dispatcher = ComputeDispatcher()
        assert dispatcher.active_unit == ComputeUnit.CPU

    def test_cpu_path_explicit(self) -> None:
        """ComputeDispatcher should route to CPU when explicitly requested."""
        dispatcher = ComputeDispatcher(compute_unit=ComputeUnit.CPU)
        assert dispatcher.active_unit == ComputeUnit.CPU

    def test_dispatch_cpu_executes_function(self) -> None:
        """dispatch() should execute the function on CPU."""

        def add(a: int, b: int) -> int:
            return a + b

        dispatcher = ComputeDispatcher(compute_unit=ComputeUnit.CPU)
        result = dispatcher.dispatch(add, 2, 3)
        assert result == 5

    def test_dispatch_with_kwargs(self) -> None:
        """dispatch() should handle keyword arguments."""

        def multiply(a: int, b: int = 2) -> int:
            return a * b

        dispatcher = ComputeDispatcher(compute_unit=ComputeUnit.CPU)
        result = dispatcher.dispatch(multiply, 5, b=3)
        assert result == 15

    def test_npu_requested_falls_back_to_cpu(self) -> None:
        """When NPU is requested, active_unit still returns CPU (not yet implemented)."""
        dispatcher = ComputeDispatcher(compute_unit=ComputeUnit.NPU)
        assert dispatcher.active_unit == ComputeUnit.CPU

    def test_dispatch_npu_falls_back_to_cpu(self) -> None:
        """dispatch() should fall back to CPU when NPU not implemented."""

        def add(a: int, b: int) -> int:
            return a + b

        dispatcher = ComputeDispatcher(compute_unit=ComputeUnit.NPU)
        result = dispatcher.dispatch(add, 10, 20)
        assert result == 30

    def test_active_unit_always_cpu(self) -> None:
        """active_unit property always returns CPU regardless of requested unit."""
        for unit in ComputeUnit:
            dispatcher = ComputeDispatcher(compute_unit=unit)
            assert dispatcher.active_unit == ComputeUnit.CPU

    def test_dispatch_preserves_return_type(self) -> None:
        """dispatch() should preserve the return type of the function."""

        def return_float() -> float:
            return 3.14

        def return_string() -> str:
            return "hello"

        dispatcher = ComputeDispatcher()
        result_float = dispatcher.dispatch(return_float)
        result_string = dispatcher.dispatch(return_string)

        assert isinstance(result_float, float)
        assert isinstance(result_string, str)

    def test_non_cpu_dispatch_logs_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """dispatch() should log a debug message when falling back from a non-CPU unit."""

        def test_fn(x: int) -> int:
            return x * 2

        dispatcher = ComputeDispatcher(compute_unit=ComputeUnit.NPU)
        with caplog.at_level(logging.DEBUG):
            result = dispatcher.dispatch(test_fn, 5)

        assert result == 10
        assert "falling back to CPU" in caplog.text

    def test_cpu_dispatch_no_debug_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """dispatch() should not log a debug fallback message when running on CPU."""

        def test_fn(x: int) -> int:
            return x * 2

        dispatcher = ComputeDispatcher(compute_unit=ComputeUnit.CPU)
        with caplog.at_level(logging.DEBUG):
            result = dispatcher.dispatch(test_fn, 5)

        assert result == 10
        assert "falling back to CPU" not in caplog.text
