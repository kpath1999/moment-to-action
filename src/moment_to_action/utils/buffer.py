"""Pre-allocated numpy buffer management.

Provides :class:`BufferSpec` and :class:`BufferPool` for zero-copy
buffer reuse on the hot inference path.
"""

from __future__ import annotations

import logging

import attrs
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@attrs.define
class BufferSpec:
    """Describes a pre-allocated output buffer."""

    shape: tuple
    """Array shape, e.g. ``(15360,)`` or ``(3, 256, 256)``."""

    dtype: npt.DTypeLike = np.float32
    """NumPy dtype for the buffer elements."""

    def allocate(self) -> np.ndarray:
        """Allocate a zeroed numpy array matching this spec."""
        return np.zeros(self.shape, dtype=self.dtype)


class BufferPool:
    """Pool of pre-allocated numpy arrays.

    Preprocessors request a buffer by spec, use it, then read the result.
    Eliminates per-inference malloc on the hot path.

    Usage::

        pool = BufferPool()
        pool.register("audio_chunk", BufferSpec((15360,), np.float32))
        buf = pool.get("audio_chunk")   # zero-copy, pre-allocated
        # ... fill buf in-place ...
        out = buf.copy()                # copy only when handing off downstream
    """

    def __init__(self) -> None:
        self._pool: dict[str, np.ndarray] = {}
        self._specs: dict[str, BufferSpec] = {}

    def register(self, name: str, spec: BufferSpec, *, overwrite: bool = False) -> None:
        """Register and pre-allocate a named buffer. Call once at init."""
        if name not in self._pool or overwrite:
            self._pool[name] = spec.allocate()
            self._specs[name] = spec
            logger.debug("BufferPool: allocated '%s' %s %s", name, spec.shape, spec.dtype)

    def get(self, name: str) -> np.ndarray:
        """Return the pre-allocated buffer by name.

        The caller fills it in-place — no allocation occurs.
        """
        if name not in self._pool:
            msg = (
                f"Buffer '{name}' not registered. "
                f"Call register() in __init__. Available: {list(self._pool)}"
            )
            raise KeyError(msg)
        return self._pool[name]

    def get_or_register(self, name: str, spec: BufferSpec) -> np.ndarray:
        """Register if needed, then return the buffer."""
        if name not in self._pool:
            self.register(name, spec)
        return self._pool[name]

    @property
    def total_bytes(self) -> int:
        """Total bytes allocated across all buffers."""
        return sum(a.nbytes for a in self._pool.values())
