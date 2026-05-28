"""Abstract base class for image models."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from moment_to_action.models._base import BaseModel

if TYPE_CHECKING:
    import numpy as np


class ImageModel(BaseModel):
    """Abstract base for models that accept image tensors.

    Subclasses implement the three-stage inference pipeline:
    ``prepare → run → post_proc``.

    The ``run`` return type is ``object`` because different detectors
    produce varying numbers of output tensors (e.g. YOLO returns three).
    Subclasses narrow the return type appropriately.
    """

    @abstractmethod
    def prepare(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a raw image frame for inference.

        Args:
            frame: Raw BGR image as ``np.ndarray`` (HxWxC, uint8).

        Returns:
            Preprocessed tensor ready to pass to :meth:`run`.
        """
        ...

    @abstractmethod
    def run(self, prepared: np.ndarray) -> object:
        """Run forward pass on a preprocessed tensor.

        Args:
            prepared: Tensor returned by :meth:`prepare`.

        Returns:
            Raw model output(s) — shape and type depend on the subclass.
        """
        ...

    @abstractmethod
    def post_proc(self, raw: object) -> list[object]:
        """Decode raw model output into structured results.

        Args:
            raw: Output returned by :meth:`run`.

        Returns:
            List of structured results (type narrowed by subclasses).
        """
        ...
