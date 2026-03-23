"""Tests for moment_to_action.utils.compute compute dispatcher."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from moment_to_action.hardware import ComputeUnit
from moment_to_action.utils.compute import ComputeDispatcher

if TYPE_CHECKING:
    from collections.abc import Callable


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

    def test_dsp_available_active_unit_returns_dsp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """active_unit should return DSP when _unit=DSP and _dsp_available=True."""
        monkeypatch.setattr(ComputeDispatcher, "_probe_dsp", lambda _: True)
        dispatcher = ComputeDispatcher(compute_unit=ComputeUnit.DSP)
        assert dispatcher.active_unit == ComputeUnit.DSP

    def test_dispatch_dsp_available_calls_dispatch_dsp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """dispatch() should call _dispatch_dsp when DSP is available."""
        monkeypatch.setattr(ComputeDispatcher, "_probe_dsp", lambda _: True)
        dispatcher = ComputeDispatcher(compute_unit=ComputeUnit.DSP)

        call_log: list[str] = []

        def mock_dispatch_dsp(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            call_log.append("_dispatch_dsp_called")
            return fn(*args, **kwargs)

        monkeypatch.setattr(dispatcher, "_dispatch_dsp", mock_dispatch_dsp)

        def add(a: int, b: int) -> int:
            return a + b

        result = dispatcher.dispatch(add, 5, 7)
        assert result == 12
        assert call_log == ["_dispatch_dsp_called"]

    def test_dispatch_dsp_logs_debug_message(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """_dispatch_dsp should log a debug message."""
        import logging

        monkeypatch.setattr(ComputeDispatcher, "_probe_dsp", lambda _: True)
        dispatcher = ComputeDispatcher(compute_unit=ComputeUnit.DSP)

        def test_fn(x: int) -> int:
            return x * 2

        with caplog.at_level(logging.DEBUG):
            result = dispatcher.dispatch(test_fn, 5)

        assert result == 10
        assert "DSP dispatch requested for test_fn — falling back to CPU" in caplog.text
