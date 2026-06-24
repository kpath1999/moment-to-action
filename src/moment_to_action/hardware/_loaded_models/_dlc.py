"""QAIRT DLC LoadedModel — shared across platforms."""

from __future__ import annotations

import contextlib
from typing import Any

from moment_to_action.hardware._loaded_model import LoadedModel
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType


class DlcModel(LoadedModel):
    """A DLC model loaded via QAIRT.

    Used on QCS6490 (HTP/NPU) and on x86_64 (CPU backend, debug only).

    Attributes:
        _unit: The compute unit this model runs on.
        _raw: The QAIRT model handle.
        _dtype: Quantization type.
        _unloaded: Whether :meth:`unload` has already been called.
    """

    def __init__(
        self,
        unit: ComputeUnit,
        raw: Any,
        dtype: DataType = DataType.W8A8,
    ) -> None:
        """Initialize a DlcModel.

        Args:
            unit: The compute unit this model runs on (NPU, CPU, etc.).
            raw: A QAIRT model handle (from ``qairt.load``).
            dtype: Quantization type — defaults to W8A8.
        """
        self._unit = unit
        self._raw = raw
        self._dtype = dtype
        self._unloaded = False

    @property
    def unit(self) -> ComputeUnit:
        """Compute unit this model is resident on."""
        return self._unit

    @property
    def dtype(self) -> DataType:
        """Quantization type of this DLC model."""
        return self._dtype

    @property
    def model_type(self) -> ModelType:
        """Model format — always DLC."""
        return ModelType.DLC

    def run(self, inputs: object) -> object:
        """Run QAIRT DLC inference.

        Args:
            inputs: ``np.ndarray`` or ``dict[str, np.ndarray]`` for multi-input
                graphs (e.g. Detectron2 ROI head ``features`` + ``proposals``).

        Returns:
            ``dict[str, np.ndarray]`` — output tensor name to array mapping.
        """
        result = self._raw(inputs=inputs)  # type: ignore[operator]
        return dict(result.data)  # type: ignore[attr-defined]

    def unload(self) -> None:
        """Destroy the QAIRT model handle and release resources."""
        if not self._unloaded:
            with contextlib.suppress(Exception):
                self._raw.destroy()  # type: ignore[attr-defined]
            self._raw = None
            self._unloaded = True
