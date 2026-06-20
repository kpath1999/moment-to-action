"""LoadedModel implementations for the macOS arm64 (Apple Silicon) platform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from moment_to_action.hardware._loaded_model import LoadedModel
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

if TYPE_CHECKING:
    import onnxruntime as ort

logger = logging.getLogger(__name__)


def _tflite_set_inputs(interp: Any, inputs: np.ndarray | dict[str, np.ndarray]) -> None:
    """Feed input tensors into a LiteRT interpreter.

    Args:
        interp: An allocated LiteRT interpreter.
        inputs: Single ndarray (single-input) or name→tensor dict (multi-input).

    Raises:
        KeyError: If a named input is not found in the model.
        TypeError: If a tensor dtype does not match the model's expected dtype.
    """
    input_details = interp.get_input_details()

    if isinstance(inputs, np.ndarray):
        interp.set_tensor(input_details[0]["index"], inputs)
        return

    name_to_detail = {d["name"]: d for d in input_details}
    for name, tensor in inputs.items():
        if name not in name_to_detail:
            available = list(name_to_detail.keys())
            msg = f"Input name {name!r} not found in model. Available: {available}"
            raise KeyError(msg)
        detail = name_to_detail[name]
        if tensor.dtype != detail["dtype"]:
            msg = (
                f"Input {name!r} dtype mismatch: "
                f"got {tensor.dtype}, model expects {detail['dtype']}"
            )
            raise TypeError(msg)
        interp.set_tensor(detail["index"], tensor)


class MacOSARM64TfliteModel(LoadedModel):
    """A TFLite model loaded on macOS arm64 CPU via LiteRT.

    Attributes:
        _interp: The underlying LiteRT interpreter.
        _dtype: Data type of this model.
        _unloaded: Whether :meth:`unload` has already been called.
    """

    def __init__(self, interp: Any, dtype: DataType = DataType.FP32) -> None:
        """Initialize a MacOSARM64TfliteModel.

        Args:
            interp: An allocated LiteRT interpreter handle.
            dtype: Data type of this model (defaults to FP32).
        """
        self._interp = interp
        self._dtype = dtype
        self._unloaded = False

    @property
    def unit(self) -> ComputeUnit:
        """Compute unit — always CPU on macOS arm64."""
        return ComputeUnit.CPU

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


class MacOSARM64ONNXModel(LoadedModel):
    """An ONNX model loaded on macOS arm64 CPU via ONNX Runtime.

    Attributes:
        _session: The underlying ``onnxruntime.InferenceSession``.
        _dtype: Data type of this model.
        _unloaded: Whether :meth:`unload` has already been called.
    """

    def __init__(self, session: Any, dtype: DataType = DataType.FP32) -> None:
        """Initialize a MacOSARM64ONNXModel.

        Args:
            session: An ``onnxruntime.InferenceSession`` handle.
            dtype: Data type of this model (defaults to FP32).
        """
        self._session = session
        self._dtype = dtype
        self._unloaded = False

    @property
    def unit(self) -> ComputeUnit:
        """Compute unit — always CPU on macOS arm64."""
        return ComputeUnit.CPU

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
