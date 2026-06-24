"""Concrete LoadedModel implementations shared across all platforms."""

from __future__ import annotations

from moment_to_action.hardware._loaded_models._dlc import DlcModel
from moment_to_action.hardware._loaded_models._llama import LlamaModel
from moment_to_action.hardware._loaded_models._onnx import OnnxModel
from moment_to_action.hardware._loaded_models._tflite import TfliteModel
from moment_to_action.hardware._loaded_models._torch import TorchModel

__all__ = [
    "DlcModel",
    "LlamaModel",
    "OnnxModel",
    "TfliteModel",
    "TorchModel",
]
