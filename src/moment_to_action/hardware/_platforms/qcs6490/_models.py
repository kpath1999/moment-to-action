"""LoadedModel implementations for the QCS6490 platform."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from moment_to_action.hardware._loaded_model import LoadedModel
from moment_to_action.hardware._platforms._shared import _tflite_set_inputs
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

if TYPE_CHECKING:
    import onnxruntime as ort

logger = logging.getLogger(__name__)


class QCS6490TfliteModel(LoadedModel):
    """A TFLite model loaded on QCS6490 (NPU via QNN delegate, GPU, or CPU).

    Attributes:
        _unit: The compute unit this model runs on.
        _dtype: Data type this model was compiled to.
        _interp: The underlying LiteRT interpreter.
        _unloaded: Whether :meth:`unload` has already been called.
    """

    def __init__(
        self,
        unit: ComputeUnit,
        interp: Any,
        dtype: DataType = DataType.FP32,
    ) -> None:
        """Initialize a QCS6490TfliteModel.

        Args:
            unit: The compute unit this model runs on (NPU, GPU, or CPU).
            interp: An allocated LiteRT interpreter handle.
            dtype: Data type of this model (defaults to FP32).
        """
        self._unit = unit
        self._dtype = dtype
        self._interp = interp
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


class QCS6490ONNXModel(LoadedModel):
    """An ONNX model loaded on QCS6490 CPU via ONNX Runtime.

    Attributes:
        _session: The underlying ``onnxruntime.InferenceSession``.
        _unloaded: Whether :meth:`unload` has already been called.
    """

    def __init__(self, session: Any, dtype: DataType = DataType.FP32) -> None:
        """Initialize a QCS6490ONNXModel.

        Args:
            session: An ``onnxruntime.InferenceSession`` handle.
            dtype: Data type of this model (defaults to FP32).
        """
        self._session = session
        self._dtype = dtype
        self._unloaded = False

    @property
    def unit(self) -> ComputeUnit:
        """Compute unit — always CPU (ONNX Runtime on QCS6490 uses CPU EP)."""
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


class QCS6490DLCModel(LoadedModel):
    """A DLC model loaded on QCS6490 via QAIRT.

    Attributes:
        _unit: The compute unit this model runs on (typically NPU).
        _dtype: Quantization type (W8A8 or W8A16).
        _raw: The QAIRT model handle.
        _unloaded: Whether :meth:`unload` has already been called.
    """

    def __init__(
        self,
        unit: ComputeUnit,
        raw: Any,
        dtype: DataType = DataType.W8A8,
    ) -> None:
        """Initialize a QCS6490DLCModel.

        Args:
            unit: The compute unit (NPU, GPU, or CPU).
            raw: A QAIRT model handle (from ``qairt.load``).
            dtype: Quantization type — defaults to W8A8.
        """
        self._unit = unit
        self._dtype = dtype
        self._raw = raw
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
        """Destroy the QAIRT model handle and release HTP resources."""
        if not self._unloaded:
            with contextlib.suppress(Exception):
                self._raw.destroy()  # type: ignore[attr-defined]
            self._raw = None
            self._unloaded = True
