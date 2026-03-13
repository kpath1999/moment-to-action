"""preprocessors/base.py

Abstract base class for all preprocessors.

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
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

from moment_to_action.edgeperceive.hardware.types import ComputeUnit

logger = logging.getLogger(__name__)

# Generic type variables
InputT  = TypeVar("InputT")   # what goes in  (AudioInput, ImageInput, ...)
OutputT = TypeVar("OutputT")  # what comes out (list[AudioChunk], ProcessedFrame, ...)


# ---------------------------------------------------------------------------
# Pre-allocated buffer pool
# ---------------------------------------------------------------------------

@dataclass
class BufferSpec:
    """Describes a pre-allocated output buffer."""
    shape: tuple
    dtype: np.dtype = np.float32

    def allocate(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=self.dtype)


class BufferPool:
    """Pool of pre-allocated numpy arrays.

    Preprocessors request a buffer by spec, use it, return it.
    Eliminates per-inference malloc on the hot path.

    Usage:
        pool = BufferPool()
        pool.register("audio_chunk", BufferSpec((15360,), np.float32))
        buf = pool.get("audio_chunk")   # zero-copy, pre-allocated
        # ... fill buf in-place ...
        out = buf.copy()                # copy only when handing off to model
    """

    def __init__(self):
        self._pool: dict[str, np.ndarray] = {}
        self._specs: dict[str, BufferSpec] = {}

    def register(self, name: str, spec: BufferSpec) -> None:
        """Register and pre-allocate a named buffer. Call once at init."""
        if name not in self._pool:
            self._pool[name] = spec.allocate()
            self._specs[name] = spec
            logger.debug(f"BufferPool: allocated '{name}' {spec.shape} {spec.dtype}")

    def get(self, name: str) -> np.ndarray:
        """Return the pre-allocated buffer by name.
        The caller fills it in-place. No allocation occurs.
        """
        if name not in self._pool:
            raise KeyError(
                f"Buffer '{name}' not registered. "
                f"Call register() in __init__. Available: {list(self._pool)}"
            )
        return self._pool[name]

    def get_or_register(self, name: str, spec: BufferSpec) -> np.ndarray:
        """Convenience: register if needed, then return."""
        if name not in self._pool:
            self.register(name, spec)
        return self._pool[name]

    @property
    def total_bytes(self) -> int:
        return sum(a.nbytes for a in self._pool.values())


# ---------------------------------------------------------------------------
# Compute dispatch
# ---------------------------------------------------------------------------

class ComputeDispatcher:
    """Routes preprocessing operations to the right compute unit.

    When compute_unit == DSP: attempts DSP dispatch, falls back to CPU.
    When compute_unit == CPU: runs directly.

    Subclasses call self._dispatch(fn, *args) instead of fn(*args).
    When a real DSP backend exists, only this class needs to change.
    """

    def __init__(self, compute_unit: ComputeUnit = ComputeUnit.CPU):
        self._unit = compute_unit
        self._dsp_available = self._probe_dsp()

    def _probe_dsp(self) -> bool:
        """Check whether DSP backend is actually available."""
        # TODO: probe Qualcomm Hexagon SDK availability
        # For now always False - DSP backend is TODO
        return False

    def dispatch(self, fn: Callable, *args, **kwargs) -> Any:
        """Run fn(*args) on the configured compute unit.
        Falls back to CPU if requested unit unavailable.
        """
        if self._unit == ComputeUnit.DSP and self._dsp_available:
            return self._dispatch_dsp(fn, *args, **kwargs)
        return fn(*args, **kwargs)   # CPU path - direct call

    def _dispatch_dsp(self, fn: Callable, *args, **kwargs) -> Any:
        """DSP dispatch path.
        TODO: wrap fn in Hexagon SDK call via ctypes/cffi.
        For now falls through to CPU.
        """
        logger.debug(f"DSP dispatch requested for {fn.__name__} - falling back to CPU")
        return fn(*args, **kwargs)

    @property
    def active_unit(self) -> ComputeUnit:
        if self._unit == ComputeUnit.DSP and self._dsp_available:
            return ComputeUnit.DSP
        return ComputeUnit.CPU


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

    def __init__(
        self,
        compute_unit: ComputeUnit = ComputeUnit.CPU,
        metrics=None,           # Optional[MetricsCollector] - avoid circular import
    ):
        self._compute_unit = compute_unit
        self._metrics = metrics
        self._buffers = BufferPool()
        self._dispatcher = ComputeDispatcher(compute_unit)

        # Let subclass register its buffers
        self._allocate_buffers()
        logger.debug(
            f"{self.__class__.__name__} init: "
            f"unit={self._dispatcher.active_unit.name} "
            f"buffers={self._buffers.total_bytes // 1024}KB"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, input: InputT) -> OutputT:
        """Validate → preprocess → report timing.
        This is what models call. Never override this - override _process.
        """
        self._validate(input)

        t_start = time.perf_counter()
        result = self._process(input)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        if self._metrics:
            self._metrics.log_event("preprocess", {
                "preprocessor": self.__class__.__name__,
                "latency_ms": elapsed_ms,
                "compute_unit": self._dispatcher.active_unit.name,
            })

        logger.debug(f"{self.__class__.__name__}.process: {elapsed_ms:.2f}ms")
        return result

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _validate(self, input: InputT) -> None:
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
    def _process(self, input: InputT) -> OutputT:
        """The actual preprocessing logic.
        Use self._dispatch(fn, *args) for ops that can run on DSP.
        Use self._buffers.get(name) for output buffers.
        """
        ...

    # ------------------------------------------------------------------
    # Protected helpers subclasses can use
    # ------------------------------------------------------------------

    def _dispatch(self, fn: Callable, *args, **kwargs) -> Any:
        """Route a computation to DSP or CPU.
        Subclasses call this instead of calling fn() directly.
        """
        return self._dispatcher.dispatch(fn, *args, **kwargs)

    @property
    def compute_unit(self) -> ComputeUnit:
        return self._dispatcher.active_unit

    @property
    def buffer_pool(self) -> BufferPool:
        return self._buffers
