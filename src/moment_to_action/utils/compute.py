"""Compute dispatch — routes preprocessing operations to CPU or DSP."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from moment_to_action.hardware import ComputeUnit

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class ComputeDispatcher:
    """Routes preprocessing operations to the right compute unit.

    When ``compute_unit == DSP``: attempts DSP dispatch, falls back to CPU.
    When ``compute_unit == CPU``: runs directly.

    Preprocessors call ``self._dispatch(fn, *args)`` instead of ``fn(*args)``
    so that DSP acceleration can be added without changing call sites.
    """

    def __init__(self, compute_unit: ComputeUnit = ComputeUnit.CPU) -> None:
        self._unit = compute_unit
        self._dsp_available = self._probe_dsp()

    def _probe_dsp(self) -> bool:
        """Check whether a DSP backend is actually available."""
        # NOTE(nvm): probe Qualcomm Hexagon SDK availability
        # DSP backend not yet implemented
        return False

    def dispatch(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Run ``fn(*args)`` on the configured compute unit.

        Falls back to CPU if the requested unit is unavailable.
        """
        if self._unit == ComputeUnit.DSP and self._dsp_available:
            return self._dispatch_dsp(fn, *args, **kwargs)
        return fn(*args, **kwargs)  # CPU path — direct call

    def _dispatch_dsp(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """DSP dispatch path (currently falls through to CPU).

        TODO: wrap fn in a Hexagon SDK call via ctypes/cffi.
        """
        logger.debug("DSP dispatch requested for %s — falling back to CPU", fn.__name__)
        return fn(*args, **kwargs)

    @property
    def active_unit(self) -> ComputeUnit:
        """Return the currently active compute unit."""
        if self._unit == ComputeUnit.DSP and self._dsp_available:
            return ComputeUnit.DSP
        return ComputeUnit.CPU
