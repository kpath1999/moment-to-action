"""Abstract base class for all runnable models."""

from __future__ import annotations

import contextlib
import logging
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from typing_extensions import Self

from moment_to_action.metrics import MetricsCollector, NullMetricsCollector, SpanType

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np

    from moment_to_action.hardware import Platform
    from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

logger = logging.getLogger(__name__)

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

    def __init__(
        self,
        variant: str,
        path: Path,
        model_type: ModelType,
        data_type: DataType,
        *,
        backends: dict[ComputeUnit, dict[str, str]],
        input_layout: str | None = None,
    ) -> None:
        """Initialize with variant name, path, model type, backend table, and input layout.

        Args:
            variant: Registry variant key (e.g. ``"default"``, ``"qcs6490"``).
            path: Path to the model weights file or variant directory.
            model_type: File format (``ModelType.ONNX``, ``ModelType.DLC``, etc.).
            data_type: Quantization type (e.g. ``DataType.W8A8``).
            backends: Mapping of compute unit to component filename dicts.
                Keys present are the supported units; ``load()`` indexes this
                with the explicit ``unit`` arg to ``load()`` to pick the artifact filenames.
            input_layout: Input tensor memory layout, ``"NCHW"`` or ``"NHWC"``,
                or ``None`` for model types that do not require a spatial layout
                (e.g. language models).
        """
        self._variant = variant
        self._path = path
        self._model_type = model_type
        self._data_type = data_type
        self._backends = backends
        self._input_layout = input_layout
        self._platform: Platform | None = None

    @property
    def path(self) -> Path:
        """Filesystem path to the model weights file (read-only)."""
        return self._path

    def _artifact_path(self, filename: str) -> Path:
        """Resolve an artifact filename to an absolute path.

        For directory-based model paths (HuggingFace variant dirs with no file
        suffix), joins ``filename`` to the variant directory.  For file-based
        paths (Ultralytics exports that are already a concrete ``.onnx`` file),
        returns the path unchanged, ignoring ``filename``.

        Args:
            filename: Artifact filename relative to the variant directory
                (e.g. ``"model.onnx"``, ``"model.npu.bin"``).

        Returns:
            Absolute path to the artifact file.
        """
        return self._path if self._path.suffix else self._path / filename

    @property
    def is_loaded(self) -> bool:
        """True if the model has been loaded onto a backend, False otherwise."""
        return self._platform is not None

    def prepare_for_conversion(self, onnx_path: Path) -> Path:
        """Return an ONNX path ready for DLC conversion.

        The default implementation returns ``onnx_path`` unchanged.  Subclasses
        override to apply graph surgery (e.g. splitting mixed-range output tensors)
        before INT8 quantization so each output gets an independent scale.

        The caller is responsible for deleting any temporary file if the returned
        path differs from ``onnx_path``.

        Args:
            onnx_path: Path to the source ONNX model.

        Returns:
            Path to the ONNX to pass to the converter — either ``onnx_path``
            unchanged or a new temporary file.
        """
        return onnx_path

    @abstractmethod
    def _prepare(self, inputs: _InputT) -> _PreparedT:
        """Preprocess raw inputs for inference.

        Args:
            inputs: Raw input to preprocess.

        Returns:
            Preprocessed data ready to pass to :meth:`_run`.
        """
        ...

    @abstractmethod
    def _run(self, prepared: _PreparedT) -> _RawOutputT:
        """Run forward pass on preprocessed inputs.

        Args:
            prepared: Output of :meth:`_prepare`.

        Returns:
            Raw model output to pass to :meth:`_post_proc`.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        ...

    @abstractmethod
    def _post_proc(self, raw: _RawOutputT) -> list[_ResultT]:
        """Decode raw model output into structured results.

        Args:
            raw: Output returned by :meth:`_run`.

        Returns:
            List of structured results (element type narrowed by subclasses).
        """
        ...

    def prepare(self, inputs: _InputT, *, metrics: MetricsCollector | None = None) -> _PreparedT:
        """Preprocess raw inputs, recording a ``MODEL_PREPROCESS`` span in *metrics*.

        Wraps :meth:`_prepare` in a :attr:`~moment_to_action.metrics.SpanType.MODEL_PREPROCESS`
        span.  When *metrics* is ``None``, a warning is logged and a
        :class:`~moment_to_action.metrics.NullMetricsCollector` is used.

        Args:
            inputs: Raw input to preprocess.
            metrics: Active collector with an open trace to record the span.

        Returns:
            Preprocessed data ready to pass to :meth:`run`.
        """
        if metrics is None:
            logger.warning(
                "%s.prepare() called without a MetricsCollector;"
                " preprocess latency will not be recorded",
                type(self).__name__,
            )
            metrics = NullMetricsCollector()
        with metrics.start_span(SpanType.MODEL_PREPROCESS, f"{type(self).__name__}.prepare"):
            return self._prepare(inputs)

    def run(self, prepared: _PreparedT, *, metrics: MetricsCollector | None = None) -> _RawOutputT:
        """Run forward pass, recording a ``MODEL_INFERENCE`` span in *metrics*.

        Wraps :meth:`_run` in a :attr:`~moment_to_action.metrics.SpanType.MODEL_INFERENCE`
        span.  When *metrics* is ``None``, a warning is logged and a
        :class:`~moment_to_action.metrics.NullMetricsCollector` is used.

        Args:
            prepared: Output of :meth:`prepare`.
            metrics: Active collector with an open trace to record the span.

        Returns:
            Raw model output to pass to :meth:`post_proc`.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if metrics is None:
            logger.warning(
                "%s.run() called without a MetricsCollector;"
                " inference latency will not be recorded",
                type(self).__name__,
            )
            metrics = NullMetricsCollector()
        with metrics.start_span(SpanType.MODEL_INFERENCE, f"{type(self).__name__}.run"):
            return self._run(prepared)

    def post_proc(
        self, raw: _RawOutputT, *, metrics: MetricsCollector | None = None
    ) -> list[_ResultT]:
        """Decode raw model output, recording a ``MODEL_POST_PROCESS`` span in *metrics*.

        Wraps :meth:`_post_proc` in a
        :attr:`~moment_to_action.metrics.SpanType.MODEL_POST_PROCESS` span.  When
        *metrics* is ``None``, a warning is logged and a
        :class:`~moment_to_action.metrics.NullMetricsCollector` is used.

        Args:
            raw: Output returned by :meth:`run`.
            metrics: Active collector with an open trace to record the span.

        Returns:
            List of structured results (element type narrowed by subclasses).
        """
        if metrics is None:
            logger.warning(
                "%s.post_proc() called without a MetricsCollector;"
                " post-process latency will not be recorded",
                type(self).__name__,
            )
            metrics = NullMetricsCollector()
        with metrics.start_span(SpanType.MODEL_POST_PROCESS, f"{type(self).__name__}.post_proc"):
            return self._post_proc(raw)

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
    def _load(self, platform: Platform, unit: ComputeUnit) -> None:
        """Load model weights and prepare for inference.

        Args:
            platform: The hardware platform to load the model onto.
            unit: The compute unit to target (e.g. ``ComputeUnit.CPU``).

        Raises:
            RuntimeError: If the model is already loaded.
            ValueError: If *unit* is not available on *backend*.
        """
        ...

    @abstractmethod
    def _unload(self) -> None:
        """Release backend resources and reset internal state.

        Safe to call when the model is not loaded (no-op).
        """
        ...

    def load(
        self, platform: Platform, unit: ComputeUnit, *, metrics: MetricsCollector | None = None
    ) -> None:
        """Load model weights, recording a ``MODEL_LOAD`` span in *metrics*.

        Wraps :meth:`_load` in a :attr:`~moment_to_action.metrics.SpanType.MODEL_LOAD`
        metrics span.  When *metrics* is ``None``, a warning is logged and a
        :class:`~moment_to_action.metrics.NullMetricsCollector` is used so the call
        is still instrumented (no-op spans).

        Args:
            platform: The hardware platform to load the model onto.
            unit: The compute unit to target (e.g. ``ComputeUnit.CPU``).
            metrics: Active collector to record the load span.  Pass the same
                collector used for the surrounding pipeline run so load latency
                appears in the same :class:`~moment_to_action.metrics.Trace`.

        Raises:
            RuntimeError: If the model is already loaded.
            ValueError: If *unit* is not available on *platform*.
        """
        if metrics is None:
            logger.warning(
                "%s.load() called without a MetricsCollector; load latency will not be recorded",
                type(self).__name__,
            )
            metrics = NullMetricsCollector()
        with metrics.start_span(SpanType.MODEL_LOAD, f"{type(self).__name__}.load"):
            self._load(platform, unit)

    def unload(self, *, metrics: MetricsCollector | None = None) -> None:
        """Release backend resources, recording a ``MODEL_UNLOAD`` span in *metrics*.

        Wraps :meth:`_unload` in a :attr:`~moment_to_action.metrics.SpanType.MODEL_UNLOAD`
        metrics span.  When *metrics* is ``None``, a warning is logged and a
        :class:`~moment_to_action.metrics.NullMetricsCollector` is used.  When *metrics*
        is provided but no trace is currently active, a transient trace is opened
        automatically.

        Args:
            metrics: Active collector to record the unload span.

        """
        if metrics is None:
            logger.warning(
                "%s.unload() called without a MetricsCollector;"
                " unload latency will not be recorded",
                type(self).__name__,
            )
            metrics = NullMetricsCollector()
        with metrics.start_span(SpanType.MODEL_UNLOAD, f"{type(self).__name__}.unload"):
            self._unload()

    @contextlib.contextmanager
    def loaded(
        self, platform: Platform, unit: ComputeUnit, *, metrics: MetricsCollector | None = None
    ) -> Iterator[Self]:
        """Context manager: load, yield self, then unload — even on exception.

        Args:
            platform: The hardware platform to load onto.
            unit: The compute unit to target.
            metrics: Collector to record ``MODEL_LOAD`` and ``MODEL_UNLOAD`` spans.
                Passed through to :meth:`load` and :meth:`unload`.

        Yields:
            This model instance, ready for inference.

        Example:
            >>> with model.loaded(platform, ComputeUnit.CPU, metrics=collector) as m:
            ...     result = m.post_proc(m.run(m.prepare(frame)))
        """
        self.load(platform, unit, metrics=metrics)
        try:
            yield self
        finally:
            self.unload(metrics=metrics)

    def __enter__(self) -> Self:
        """Return self for use in a ``with`` block.

        :meth:`load` must be called before entering the block; :meth:`unload`
        is called automatically by :meth:`__exit__`.

        Returns:
            This model instance.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Call :meth:`_unload` on exit from the ``with`` block."""
        self._unload()

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
            self._unload()
