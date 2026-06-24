"""ONNX Runtime LoadedModel — shared across all platforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from moment_to_action.hardware._loaded_model import LoadedModel
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

if TYPE_CHECKING:
    import onnxruntime as ort


class OnnxModel(LoadedModel):
    """An ONNX model loaded on any compute unit via ONNX Runtime.

    Attributes:
        _unit: The compute unit this model runs on.
        _session: The underlying ``onnxruntime.InferenceSession``.
        _dtype: Data type of this model.
        _unloaded: Whether :meth:`unload` has already been called.
    """

    def __init__(
        self,
        unit: ComputeUnit,
        session: Any,
        dtype: DataType = DataType.FP32,
    ) -> None:
        """Initialize an OnnxModel.

        Args:
            unit: The compute unit this model runs on.
            session: An ``onnxruntime.InferenceSession`` handle.
            dtype: Data type of this model (defaults to FP32).
        """
        self._unit = unit
        self._session = session
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
        """Model format — always ONNX."""
        return ModelType.ONNX

    def run(self, inputs: object) -> object:
        """Run ONNX inference.

        Args:
            inputs: ``np.ndarray`` (single-input) or ``dict[str, np.ndarray]``
                (multi-input, keyed by ONNX input name).

        Returns:
            ``list[np.ndarray]`` — one array per output slot.
        """
        session = cast("ort.InferenceSession", self._session)
        input_details = session.get_inputs()
        if isinstance(inputs, np.ndarray):
            feed = {input_details[0].name: inputs}
        else:
            feed = cast("dict[str, np.ndarray]", inputs)
        return session.run(None, feed)

    def unload(self) -> None:
        """Release the ONNX session."""
        if not self._unloaded:
            self._session = None
            self._unloaded = True
