"""Model formats.

This is a seperate file so it can be imported by both _sources and _models without circular imports.
"""

from enum import Enum, auto


class ModelFormat(Enum):
    """Enumeration of supported model formats."""

    # Currently, we only support ONNX and the DLC format used by QAIRT,
    # but we can easily add more formats in the future as needed.
    ONNX = auto()
    DLC = auto()  # Used by QAIRT
    GGUF = auto()  # Used by llama-server
