"""Abstract base class for all preprocessors.

What lives here (truly common across all modalities):
- Compute backend reference + dispatch routing (CPU vs DSP)
- Pre-allocated output buffer management
- process() signature contract
- Validation before processing
- Timing via MetricsCollector

What does NOT live here (stays in subclasses):
- The actual operations (resampling, resizing - completely different)
- Output type (AudioChunk vs ProcessedFrame - enforced via Generic)
- Configuration (sample_rate vs image_size)

Design: Generic[InputT, OutputT]
  AudioPreprocessor(BasePreprocessor[AudioInput, list[AudioChunk]])
  ImagePreprocessor(BasePreprocessor[ImageInput, ProcessedFrame])
  VideoPreprocessor(BasePreprocessor[VideoInput, list[ProcessedFrame]])

This means the caller always knows the return type statically.
No isinstance checks downstream.

Compute dispatch:
  Subclasses call self._dispatch(fn, *args) instead of fn(*args).
  The base class routes to DSP or CPU based on ComputeUnit config.
  When DSP backend is implemented, subclasses get it for free.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, ParamSpec, TypeVar

from moment_to_action.hardware import ComputeUnit
from moment_to_action.metrics._collector import NullMetricsCollector
from moment_to_action.metrics._types import SpanType
from moment_to_action.utils import BufferPool, ComputeDispatcher

if TYPE_CHECKING:
    from collections.abc import Callable

    from moment_to_action.metrics._collector import MetricsCollector

logger = logging.getLogger(__name__)

# Generic type variables
InputT = TypeVar("InputT")  # what goes in  (AudioInput, ImageInput, ...)
OutputT = TypeVar("OutputT")  # what comes out (list[AudioChunk], ProcessedFrame, ...)
_P = ParamSpec("_P")
_R = TypeVar("_R")


# ---------------------------------------------------------------------------
# Base preprocessor
# ---------------------------------------------------------------------------


class BasePreprocessor(ABC, Generic[InputT, OutputT]):
    """Abstract base for all preprocessors.

    Subclasses implement:
        _validate(input) → None        raise ValueError if input is bad
        _allocate_buffers() → None     call self._buffers.register() for each buffer
        _process(input) → OutputT      the actual preprocessing logic

    Subclasses get for free:
        self._buffers    BufferPool    pre-allocated output buffers
        self._dispatch   callable      routes ops to CPU or DSP
        self.process()                 validates → times → _process → reports

    Example:
        class AudioPreprocessor(BasePreprocessor[AudioInput, list[AudioChunk]]):
            def _validate(self, input):
                if input.sample_rate <= 0:
                    raise ValueError("bad sample rate")

            def _allocate_buffers(self):
                self._buffers.register(
                    "chunk",
                    BufferSpec((YAMNET_NUM_SAMPLES,), np.float32)
                )

            def _process(self, input) -> list[AudioChunk]:
                buf = self._buffers.get("chunk")
                # fill buf in-place via self._dispatch(self._resample, ...)
                ...
    """

    def __init__(self, compute_unit: ComputeUnit = ComputeUnit.CPU) -> None:
        self._compute_unit = compute_unit
        self._buffers = BufferPool()
        self._dispatcher = ComputeDispatcher(compute_unit)

        # Let subclass register its buffers
        self._allocate_buffers()
        logger.debug(
            "%s init: unit=%s buffers=%dKB",
            self.__class__.__name__,
            self._dispatcher.active_unit.name,
            self._buffers.total_bytes // 1024,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, data: InputT, metrics: MetricsCollector | None = None) -> OutputT:
        """Validate, preprocess, and report timing.

        This is what models call. Never override this - override _process.

        Args:
            data:   Input data to preprocess (AudioInput, ImageInput, etc).
            metrics: MetricsCollector for reporting timing. Optional, but recommended.
                     If not provided, timing will be skipped but processing will still work.
        """
        # Ensure we always have a metrics collector to avoid null checks in subclass code
        #
        # This is done here so that preprocessors can be used standalone, outside of a pipeline,
        # if desired. Useful for testing
        if metrics is None:
            metrics = NullMetricsCollector()

        # Validate input before processing
        self._validate(data)

        # Time the processing and report to metrics
        with metrics.start_span(SpanType.MODEL_PREPROCESS, self.__class__.__name__) as span:
            span_id = span.id_  # save so we can report after processing

            # Run the actual preprocessing logic implemented by the subclass
            result = self._process(data)

        # Get elapsed time from metrics for logging
        elapsed_ms = metrics.get_span(span_id).latency_ms
        logger.debug("%s.process: %.2fms", self.__class__.__name__, elapsed_ms)

        return result

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _validate(self, data: InputT) -> None:
        """Validate input before processing.

        Raise ValueError with a descriptive message if invalid.
        Called automatically by process() before _process().
        """
        ...

    @abstractmethod
    def _allocate_buffers(self) -> None:
        """Pre-allocate output buffers via self._buffers.register().

        Called once at __init__ time - never on the hot path.
        """
        ...

    @abstractmethod
    def _process(self, data: InputT) -> OutputT:
        """The actual preprocessing logic.

        Use self._dispatch(fn, *args) for ops that can run on DSP.
        Use self._buffers.get(name) for output buffers.
        """
        ...

    # ------------------------------------------------------------------
    # Protected helpers subclasses can use
    # ------------------------------------------------------------------

    def _dispatch(self, fn: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Route a computation to DSP or CPU.

        Subclasses call this instead of calling fn() directly.
        """
        return self._dispatcher.dispatch(fn, *args, **kwargs)

    @property
    def compute_unit(self) -> ComputeUnit:
        """Return the active compute unit."""
        return self._dispatcher.active_unit

    @property
    def buffer_pool(self) -> BufferPool:
        """Return the pre-allocated buffer pool."""
        return self._buffers
