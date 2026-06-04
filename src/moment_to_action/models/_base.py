"""Abstract base class for all runnable models."""

from __future__ import annotations

import contextlib
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np

    from moment_to_action.hardware import ComputeBackend

_InputT = TypeVar("_InputT")
_PreparedT = TypeVar("_PreparedT")
_RawOutputT = TypeVar("_RawOutputT")
_ResultT = TypeVar("_ResultT")


class BaseModel(ABC, Generic[_InputT, _PreparedT, _RawOutputT, _ResultT]):
    """Abstract base for all loadable, runnable models.

    Every model follows a three-stage inference pipeline::

        prepare(inputs) -> PreparedT
        run(prepared)   -> RawOutputT
        post_proc(raw)  -> list[ResultT]

    Plus a ``verify_outputs`` method for correctness checking against reference
    data.  Use :meth:`loaded` as a context manager to pair load/unload:

    Example:
        >>> with model_mgr.get_model(ModelID.YOLO_V8).loaded(backend) as m:
        ...     results = m.post_proc(m.run(m.prepare(frame)))

    Type parameters:
        _InputT: Raw input type accepted by :meth:`prepare`.
        _PreparedT: Output of :meth:`prepare` / input to :meth:`run`.
        _RawOutputT: Output of :meth:`run` / input to :meth:`post_proc`.
        _ResultT: Element type returned by :meth:`post_proc`.

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
    def path(self) -> Path:
        """Filesystem path to the model weights file (read-only)."""
        return self._path

    @property
    def is_loaded(self) -> bool:
        """True if the model has been loaded onto a backend, False otherwise."""
        return self._backend is not None

    @abstractmethod
    def prepare(self, inputs: _InputT) -> _PreparedT:
        """Preprocess raw inputs for inference.

        Args:
            inputs: Raw input to preprocess.

        Returns:
            Preprocessed data ready to pass to :meth:`run`.
        """
        ...

    @abstractmethod
    def run(self, prepared: _PreparedT) -> _RawOutputT:
        """Run forward pass on preprocessed inputs.

        Args:
            prepared: Output of :meth:`prepare`.

        Returns:
            Raw model output to pass to :meth:`post_proc`.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        ...

    @abstractmethod
    def post_proc(self, raw: _RawOutputT) -> list[_ResultT]:
        """Decode raw model output into structured results.

        Args:
            raw: Output returned by :meth:`run`.

        Returns:
            List of structured results (element type narrowed by subclasses).
        """
        ...

    @abstractmethod
    def verify_outputs(
        self,
        inputs: np.ndarray,
        ref_outputs: list[np.ndarray],
        *,
        tol: float,
        is_npu: bool,
    ) -> tuple[bool, str]:
        """Verify model outputs against reference data.

        Args:
            inputs: Input array of shape ``(N, ...)``.
            ref_outputs: List of reference output arrays, each of shape ``(N, ...)``.
            tol: Max absolute element-wise error threshold for raw comparison.
            is_npu: When True, skip raw diff and compare decoded outputs only.

        Returns:
            ``(passed, fail_reason)``.  ``passed`` is True when all samples
            pass; ``fail_reason`` is empty on success or describes the first
            failure.
        """
        ...

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
            ...     result = m.post_proc(m.run(m.prepare(frame)))
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
