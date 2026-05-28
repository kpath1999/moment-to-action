"""Abstract base class for all runnable models."""

from __future__ import annotations

import contextlib
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from moment_to_action.hardware import ComputeBackend


class BaseModel(ABC):
    """Abstract base for all loadable, runnable models.

    Subclasses implement :meth:`load` and :meth:`unload` for lifecycle management.
    Image-specific inference methods (``prepare``, ``run``, ``post_proc``) are
    declared on :class:`~moment_to_action.models.image.ImageModel`.

    Use :meth:`loaded` as a context manager to automatically pair load/unload:

    Example:
        >>> with model_mgr.get_model(ModelID.YOLO_V8).loaded(backend) as model:
        ...     detections = model.decode(model.run(model.prepare(frame)), frame.shape[:2])

    Args:
        variant: Variant name used to identify this instance in the registry.
        path: Filesystem path to the model weights file.
    """

    def __init__(self, variant: str, path: Path) -> None:
        """Initialize with variant name and path.

        Args:
            variant: Registry variant key (e.g. ``"default"``, ``"qcs6490"``).
            path: Path to the model weights file.
        """
        self._variant = variant
        self._path = path
        self._backend: ComputeBackend | None = None

    @property
    def is_loaded(self) -> bool:
        """True if the model has been loaded onto a backend, False otherwise."""
        return self._backend is not None

    @abstractmethod
    def load(self, backend: ComputeBackend) -> None:
        """Load model weights and prepare for inference.

        Args:
            backend: The hardware backend to load the model onto.

        Raises:
            RuntimeError: If the model is already loaded.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release backend resources and reset internal state.

        Safe to call when the model is not loaded (no-op).
        """
        ...

    @contextlib.contextmanager
    def loaded(self, backend: ComputeBackend) -> Iterator[Self]:
        """Context manager: load, yield self, then unload — even on exception.

        Args:
            backend: The hardware backend to load onto.

        Yields:
            This model instance, ready for inference.

        Example:
            >>> with model.loaded(backend) as m:
            ...     result = m.run(m.prepare(frame))
        """
        self.load(backend)
        try:
            yield self
        finally:
            self.unload()

    def __enter__(self) -> Self:
        """Return self for use in a ``with`` block.

        :meth:`load` must be called before entering the block; :meth:`unload`
        is called automatically by :meth:`__exit__`.

        Returns:
            This model instance.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Call :meth:`unload` on exit from the ``with`` block."""
        self.unload()

    def __del__(self) -> None:
        """Warn and unload if still loaded when garbage-collected.

        A loaded model being GC-collected indicates a missing :meth:`unload`
        call or :meth:`loaded` context manager.  A :exc:`ResourceWarning` is
        emitted (same convention as file handles and sockets) and unload is
        attempted as a best-effort cleanup.
        """
        if not self.is_loaded:
            return
        warnings.warn(
            f"{type(self).__name__} garbage-collected while still loaded; "
            "call unload() explicitly or use the loaded() context manager",
            ResourceWarning,
            stacklevel=2,
        )
        with contextlib.suppress(Exception):
            self.unload()
