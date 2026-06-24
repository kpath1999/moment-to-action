"""LoadedModel implementations for the x86_64 platform."""

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


class X86_64TfliteModel(LoadedModel):  # noqa: N801
    """A TFLite model loaded on x86_64 CPU via LiteRT/XNNPACK.

    Attributes:
        _interp: The underlying LiteRT interpreter.
        _dtype: Data type of this model.
        _unloaded: Whether :meth:`unload` has already been called.
    """

    def __init__(self, interp: Any, dtype: DataType = DataType.FP32) -> None:
        """Initialize an X86_64TfliteModel.

        Args:
            interp: An allocated LiteRT interpreter handle.
            dtype: Data type of this model (defaults to FP32).
        """
        self._interp = interp
        self._dtype = dtype
        self._unloaded = False

    @property
    def unit(self) -> ComputeUnit:
        """Compute unit — always CPU on x86_64."""
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


class X86_64ONNXModel(LoadedModel):  # noqa: N801
    """An ONNX model loaded on x86_64 CPU via ONNX Runtime.

    Attributes:
        _session: The underlying ``onnxruntime.InferenceSession``.
        _dtype: Data type of this model.
        _unloaded: Whether :meth:`unload` has already been called.
    """

    def __init__(self, session: Any, dtype: DataType = DataType.FP32) -> None:
        """Initialize an X86_64ONNXModel.

        Args:
            session: An ``onnxruntime.InferenceSession`` handle.
            dtype: Data type of this model (defaults to FP32).
        """
        self._session = session
        self._dtype = dtype
        self._unloaded = False

    @property
    def unit(self) -> ComputeUnit:
        """Compute unit — always CPU on x86_64."""
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


class X86_64DLCModel(LoadedModel):  # noqa: N801
    """A DLC model loaded on x86_64 via QAIRT (CPU backend, debug only).

    Enables running QCS6490 DLC models locally without a device for debugging.
    Requires the QAIRT SDK to be installed (``m2a qairt install``).

    Attributes:
        _raw: The QAIRT model handle.
        _dtype: Data type / quantization of this model.
        _unloaded: Whether :meth:`unload` has already been called.
    """

    def __init__(self, raw: Any, dtype: DataType = DataType.FP32) -> None:
        """Initialize an X86_64DLCModel.

        Args:
            raw: A QAIRT model handle (from ``qairt.load``).
            dtype: Data type / quantization of this model (defaults to FP32).
        """
        self._raw = raw
        self._dtype = dtype
        self._unloaded = False

    @property
    def unit(self) -> ComputeUnit:
        """Compute unit — always CPU on x86_64."""
        return ComputeUnit.CPU

    @property
    def dtype(self) -> DataType:
        """Data type / quantization of this DLC model."""
        return self._dtype

    @property
    def model_type(self) -> ModelType:
        """Model format — always DLC."""
        return ModelType.DLC

    def run(self, inputs: object) -> object:
        """Run QAIRT DLC inference on CPU.

        Args:
            inputs: ``np.ndarray`` or ``dict[str, np.ndarray]`` for multi-input
                graphs.

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
