"""Tests for moment_to_action.utils.compute compute dispatcher."""

from __future__ import annotations

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

    def test_probe_dsp_returns_false(self) -> None:
        """_probe_dsp() should always return False (no DSP on laptop)."""
        dispatcher = ComputeDispatcher()
        assert dispatcher._probe_dsp() is False

    def test_dsp_requested_falls_back_to_cpu(self) -> None:
        """When DSP is requested but unavailable, should fall back to CPU."""
        dispatcher = ComputeDispatcher(compute_unit=ComputeUnit.DSP)
        # _probe_dsp() returns False, so DSP is not available
        assert dispatcher.active_unit == ComputeUnit.CPU

    def test_dispatch_dsp_falls_back_to_cpu(self) -> None:
        """dispatch() should fall back to CPU when DSP is unavailable."""

        def add(a: int, b: int) -> int:
            return a + b

        dispatcher = ComputeDispatcher(compute_unit=ComputeUnit.DSP)
        result = dispatcher.dispatch(add, 10, 20)
        assert result == 30

    def test_active_unit_property(self) -> None:
        """active_unit property should return the actual active compute unit."""
        dispatcher_cpu = ComputeDispatcher(compute_unit=ComputeUnit.CPU)
        assert dispatcher_cpu.active_unit == ComputeUnit.CPU

        dispatcher_dsp = ComputeDispatcher(compute_unit=ComputeUnit.DSP)
        # Should fall back to CPU since DSP is unavailable
        assert dispatcher_dsp.active_unit == ComputeUnit.CPU

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
