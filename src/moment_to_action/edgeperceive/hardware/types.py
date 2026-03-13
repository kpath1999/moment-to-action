"""hardware/types.py

Hardware-related enums. Kept separate from compute_backend.py
to avoid circular imports — stages and preprocessors import
ComputeUnit from here, not from compute_backend.py.
"""

from enum import Enum, auto


class ComputeUnit(Enum):
    """Available compute units on the QCS6490."""

    CPU = auto()
    NPU = auto()
    GPU = auto()
    DSP = auto()
