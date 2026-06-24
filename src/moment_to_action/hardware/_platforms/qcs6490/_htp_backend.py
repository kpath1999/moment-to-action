"""QCS6490 HTP (Hexagon) NPU backend — DLC via QAIRT + TFLite via QNN delegate."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from moment_to_action.hardware._backend import ComputeBackend
from moment_to_action.hardware._platforms._shared import _load_litert_interpreter
from moment_to_action.hardware._platforms.qcs6490._models import (
    QCS6490DLCModel,
    QCS6490TfliteModel,
)
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

if TYPE_CHECKING:
    from moment_to_action.hardware._loaded_model import LoadedModel

logger = logging.getLogger(__name__)

# Path to the Qualcomm QNN TFLite delegate shared library on-device.
_QNN_DELEGATE_PATH = "/usr/lib/libQnnTFLiteDelegate.so"

# QAIRT backend string for HTP/NPU.
_QAIRT_HTP_BACKEND = "HTP"


def _load_litert_delegate() -> list:
    """Load the QNN TFLite delegate for the Hexagon HTP.

    Returns:
        A single-element list containing the loaded delegate object.

    Raises:
        RuntimeError: If the delegate shared library cannot be loaded.
    """
    try:
        from ai_edge_litert.interpreter import load_delegate as _ld  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from tensorflow.lite.python.interpreter import load_delegate as _ld  # noqa: PLC0415

    try:
        delegate = _ld(_QNN_DELEGATE_PATH)
    except Exception as e:
        msg = f"QNN HTP delegate unavailable at {_QNN_DELEGATE_PATH!r}: {e}"
        raise RuntimeError(msg) from e
    logger.info("QNN HTP delegate loaded from %s", _QNN_DELEGATE_PATH)
    return [delegate]


class QCS6490HTPBackend(ComputeBackend):
    """NPU inference backend for the QCS6490 Hexagon HTP.

    Loads DLC models via QAIRT and TFLite models via the QNN TFLite delegate.
    DLC models are initialized with the HTP (``"HTP"``) backend.

    Raises:
        RuntimeError: At construction time if the QNN TFLite delegate cannot
            be loaded (i.e. the device libraries are absent).
    """

    _SUPPORTED_DTYPES: frozenset[DataType] = frozenset({DataType.W8A8, DataType.W8A16})
    _SUPPORTED_FORMATS: frozenset[ModelType] = frozenset({ModelType.DLC, ModelType.TFLITE})

    def __init__(self) -> None:
        """Initialize the HTP backend.

        Eagerly loads the QNN TFLite delegate at construction time so that
        delegate failures are detected early rather than at first inference.

        Raises:
            RuntimeError: If the QNN TFLite delegate cannot be loaded.
        """
        self._delegates = _load_litert_delegate()
        logger.info("QCS6490HTPBackend: initialized (QNN delegate loaded)")

    @property
    def unit(self) -> ComputeUnit:
        """The compute unit — always NPU."""
        return ComputeUnit.NPU

    @property
    def supported_dtypes(self) -> set[DataType]:
        """Supported data types: W8A8 and W8A16."""
        return set(self._SUPPORTED_DTYPES)

    @property
    def supported_formats(self) -> set[ModelType]:
        """Supported formats: DLC and TFLITE."""
        return set(self._SUPPORTED_FORMATS)

    def load_tflite(self, path: str | os.PathLike[str]) -> LoadedModel:
        """Load a TFLite model and run it on the Hexagon HTP via QNN delegate.

        Args:
            path: Path to the ``.tflite`` model file.

        Returns:
            A :class:`~_models.QCS6490TfliteModel` backed by the QNN delegate.

        Raises:
            RuntimeError: If the delegate fails to apply to this model.
        """
        p = os.fspath(path)
        interp = _load_litert_interpreter(p, self._delegates)
        logger.info("QCS6490HTPBackend: loaded %s on NPU", p)
        return QCS6490TfliteModel(unit=ComputeUnit.NPU, interp=interp)

    def load_dlc(self, path: str | os.PathLike[str]) -> LoadedModel:
        """Load a DLC model and initialize it on the Hexagon HTP via QAIRT.

        Args:
            path: Path to the ``.dlc`` model file.

        Returns:
            A :class:`~_models.QCS6490DLCModel` initialized on the HTP.

        Raises:
            RuntimeError: If the QAIRT SDK is not available on this device.
        """
        try:
            import qairt  # noqa: PLC0415
        except Exception as exc:
            msg = "QAIRT SDK is not available; load_dlc requires a QCS6490 device"
            raise RuntimeError(msg) from exc
        raw = qairt.load(os.fspath(path))
        raw.initialize(backend=_QAIRT_HTP_BACKEND)
        logger.info("QCS6490HTPBackend: loaded DLC %s on HTP", path)
        return QCS6490DLCModel(unit=ComputeUnit.NPU, raw=raw)
