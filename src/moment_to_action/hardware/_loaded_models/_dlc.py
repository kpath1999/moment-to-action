"""QAIRT DLC LoadedModel — shared across platforms."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from moment_to_action.hardware._loaded_model import LoadedModel
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

if TYPE_CHECKING:
    from qairt import Model


class DlcModel(LoadedModel):
    """A DLC model loaded via QAIRT.

    Used on QCS6490 (HTP/NPU) and on x86_64 (CPU backend, debug only).

    Attributes:
        _unit: The compute unit this model runs on.
        _raw: The QAIRT model handle.
        _dtype: Quantization type.
    """

    def __init__(
        self,
        unit: ComputeUnit,
        raw: Model,
        dtype: DataType,
    ) -> None:
        """Initialize a DlcModel.

        Args:
            unit: The compute unit this model runs on (NPU, CPU, etc.).
            raw: A QAIRT model handle (from ``qairt.load``).
            dtype: Quantization type.
        """
        self._unit = unit
        self._raw: Model | None = raw
        self._dtype = dtype

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

        Raises:
            RuntimeError: If :meth:`unload` has already been called.
        """
        if self._raw is None:
            msg = "DlcModel has been unloaded"
            raise RuntimeError(msg)

        result = self._raw(inputs=inputs)
        return dict(result.data)

    def unload(self) -> None:
        """Destroy the QAIRT model handle and release resources."""
        if self._raw is None:
            # Nothing to do, already unloaded
            return

        # Do the unload
        with contextlib.suppress(Exception):
            self._raw.destroy()
        self._raw = None

    @property
    def _unloaded(self) -> bool:
        """Check if this model has already been unloaded."""
        return self._raw is None
