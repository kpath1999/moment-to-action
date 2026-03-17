"""Compute dispatch — routes preprocessing operations to CPU or DSP."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from moment_to_action.hardware import ComputeUnit

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two vectors, safe against zero-norm inputs."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


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
