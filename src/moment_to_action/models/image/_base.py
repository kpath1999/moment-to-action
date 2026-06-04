"""Abstract base class for image models."""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

import numpy as np

from moment_to_action.models._base import BaseModel

_RawOutputT = TypeVar("_RawOutputT")
_ResultT = TypeVar("_ResultT")


class ImageModel(
    BaseModel[np.ndarray, np.ndarray, _RawOutputT, _ResultT],
    Generic[_RawOutputT, _ResultT],
):
    """Abstract base for models that accept image tensors.

    Fixes :class:`~moment_to_action.models.BaseModel` ``_InputT`` and
    ``_PreparedT`` to ``np.ndarray``, so :meth:`prepare` takes and returns
    numpy arrays.  The raw output type (``_RawOutputT``) and structured result
    type (``_ResultT``) remain free for subclasses to fix.

    Type parameters:
        _RawOutputT: Output of :meth:`run` / input to :meth:`post_proc`.
        _ResultT: Element type returned by :meth:`post_proc`.
    """

    @abstractmethod
    def prepare(self, inputs: np.ndarray) -> np.ndarray:
        """Preprocess a raw image frame for inference.

        Args:
            inputs: Raw BGR image as ``np.ndarray`` (HxWxC, uint8).

        Returns:
            Preprocessed tensor ready to pass to :meth:`run`.
        """
        ...

    @abstractmethod
    def run(self, prepared: np.ndarray) -> _RawOutputT:
        """Run forward pass on a preprocessed tensor.

        Args:
            prepared: Tensor returned by :meth:`prepare`.

        Returns:
            Raw model output(s) to pass to :meth:`post_proc`.
        """
        ...

    @abstractmethod
    def post_proc(self, raw: _RawOutputT) -> list[_ResultT]:
        """Decode raw model output into structured results.

        Args:
            raw: Output returned by :meth:`run`.

        Returns:
            List of structured results (type narrowed by subclasses).
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
            inputs: Input array of shape ``(N, C, H, W)``.
            ref_outputs: List of reference output arrays, each of shape ``(N, ...)``.
            tol: Max absolute element-wise error for raw comparison.
            is_npu: When True, skip raw diff and compare decoded outputs only.

        Returns:
            ``(passed, fail_reason)``.
        """
        ...
