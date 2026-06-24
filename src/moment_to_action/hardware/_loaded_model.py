"""LoadedModel — abstract base for models resident in memory on a compute unit."""

from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Generator

    from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

logger = logging.getLogger(__name__)


class LoadedModel(ABC):
    """A model loaded into memory on a specific compute unit.

    Concrete subclasses live in ``hardware/_platforms/<chip>/_models.py`` —
    one subclass per runtime (TFLite, ONNX, DLC, …).

    Use as a context manager or call :meth:`unload` explicitly to release
    resources.  :meth:`__del__` calls :meth:`unload` as a GC safety net.
    """

    @property
    @abstractmethod
    def unit(self) -> ComputeUnit:
        """Compute unit this model is resident on.

        Returns:
            The ``ComputeUnit`` enum member for this model's execution target.
        """
        ...

    @property
    @abstractmethod
    def dtype(self) -> DataType:
        """Data / quantization type of this model.

        Returns:
            The ``DataType`` enum member (e.g. ``FP32``, ``W8A8``).
        """
        ...

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """File format this model was loaded from.

        Returns:
            The ``ModelType`` enum member (e.g. ``TFLITE``, ``DLC``).
        """
        ...

    @abstractmethod
    def run(self, inputs: object) -> object:
        """Run inference.

        Callers are responsible for casting inputs to the appropriate type
        (``np.ndarray``, ``dict[str, np.ndarray]``, etc.) and for casting
        the return value to the expected output type.

        Args:
            inputs: Input data — format is runtime-specific.

        Returns:
            Output data — format is runtime-specific.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model resources.

        May be called multiple times safely — implementations must be
        idempotent.
        """
        ...

    def __enter__(self) -> Self:
        """Enter context manager — returns self.

        Returns:
            This ``LoadedModel`` instance.
        """
        return self

    def __exit__(self, *_: object) -> None:
        """Exit context manager — calls :meth:`unload`.

        Args:
            *_: Ignored exception info.
        """
        self.unload()

    def __del__(self) -> None:
        """GC safety net — calls :meth:`unload`, swallowing all exceptions."""
        with contextlib.suppress(Exception):
            self.unload()


class LoadedStreamableModel(LoadedModel):
    """A :class:`LoadedModel` that additionally supports token-by-token streaming.

    Subclasses must implement :meth:`stream` in addition to all abstract
    members from :class:`LoadedModel`.
    """

    @abstractmethod
    def stream(self, inputs: object) -> Generator[str, None, None]:
        """Stream inference output token by token.

        Args:
            inputs: Input data — format is runtime-specific.

        Yields:
            String tokens/chunks as they are produced by the model.
        """
        ...
