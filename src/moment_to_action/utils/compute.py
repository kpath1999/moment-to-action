"""Compute dispatch — routes preprocessing operations to CPU or NPU."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from moment_to_action.hardware import ComputeUnit

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


class ComputeDispatcher:
    """Routes preprocessing operations to the right compute unit.

    When ``compute_unit == CPU``: runs directly.
    When ``compute_unit == NPU``: NPU-accelerated dispatch is not yet
    implemented and falls back to CPU transparently.

    Preprocessors call ``self._dispatch(fn, *args)`` instead of ``fn(*args)``
    so that hardware-accelerated paths can be added without changing call sites.
    """

    def __init__(self, compute_unit: ComputeUnit = ComputeUnit.CPU) -> None:
        """Initialize the dispatcher with the target compute unit.

        Args:
            compute_unit: The compute unit to target (defaults to CPU).
        """
        self._unit = compute_unit

    def dispatch(self, fn: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Run ``fn(*args)`` on the configured compute unit.

        Falls back to CPU for any unit without a native implementation.

        Args:
            fn: Function to dispatch.
            *args: Positional arguments forwarded to ``fn``.
            **kwargs: Keyword arguments forwarded to ``fn``.

        Returns:
            Return value of ``fn(*args, **kwargs)``.
        """
        if self._unit != ComputeUnit.CPU:
            logger.debug("Non-CPU dispatch requested for %s — falling back to CPU", fn.__name__)
        return fn(*args, **kwargs)

    @property
    def active_unit(self) -> ComputeUnit:
        """Return the currently active compute unit (always CPU for now).

        Returns:
            ``ComputeUnit.CPU`` — all dispatch paths currently run on CPU.
        """
        return ComputeUnit.CPU
