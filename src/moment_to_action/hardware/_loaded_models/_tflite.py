"""TFLite LoadedModel — shared across all platforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from moment_to_action.hardware._loaded_model import LoadedModel
from moment_to_action.hardware._platforms._shared import _tflite_set_inputs
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

if TYPE_CHECKING:
    import numpy as np


class TfliteModel(LoadedModel):
    """A TFLite model loaded on any compute unit via LiteRT.

    Attributes:
        _unit: The compute unit this model runs on.
        _interp: The underlying LiteRT interpreter.
        _dtype: Data type of this model.
        _unloaded: Whether :meth:`unload` has already been called.
    """

    def __init__(
        self,
        unit: ComputeUnit,
        interp: Any,
        dtype: DataType,
    ) -> None:
        """Initialize a TfliteModel.

        Args:
            unit: The compute unit this model runs on (CPU, GPU, or NPU).
            interp: An allocated LiteRT interpreter handle.
            dtype: Data type of this model.
        """
        self._unit = unit
        self._interp = interp
        self._dtype = dtype
        self._unloaded = False

    @property
    def unit(self) -> ComputeUnit:
        """Compute unit this model is resident on."""
        return self._unit

    @property
    def dtype(self) -> DataType:
        """Data type this model was compiled to."""
        return self._dtype

    @property
    def model_type(self) -> ModelType:
        """Model format — always TFLITE."""
        return ModelType.TFLITE

    def run(self, inputs: object) -> object:
        """Run TFLite inference.

        Args:
            inputs: ``np.ndarray`` (single-input) or ``dict[str, np.ndarray]``
                (multi-input).

        Returns:
            ``list[np.ndarray]`` — one array per output slot.
        """
        interp = self._interp
        _tflite_set_inputs(interp, cast("np.ndarray | dict[str, np.ndarray]", inputs))
        interp.invoke()
        return [interp.get_tensor(d["index"]) for d in interp.get_output_details()]

    def unload(self) -> None:
        """Release the interpreter handle."""
        if not self._unloaded:
            self._interp = None
            self._unloaded = True
