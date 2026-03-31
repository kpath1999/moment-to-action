"""Hardware type definitions — enums and data models.

Kept separate to avoid circular imports: stages and preprocessors
import ``ComputeUnit`` from here, not from the backend module.
"""

from __future__ import annotations

from enum import StrEnum

import attrs


class ComputeUnit(StrEnum):
    """Available compute units on a hardware accelerator platform."""

    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"
    DSP = "DSP"


@attrs.frozen
class PowerSample:
    """A single power measurement snapshot for one compute unit."""

    timestamp: float
    """Unix timestamp of the measurement (seconds)."""

    device: ComputeUnit
    """The device that was active during sampling."""

    power_mw: float
    """Power draw in milliwatts."""

    utilization_pct: float
    """Utilisation percentage (0-100)."""


@attrs.frozen
class TorchExecutionPolicy:
    """Resolved torch execution configuration for model loading and inference.

    Attributes:
        device: Torch device string (for example ``"cpu"``, ``"cuda"``, ``"mps"``).
        dtype: Torch dtype attribute name (for example ``"float32"``, ``"float16"``).
    """

    device: str
    """Resolved torch device string."""

    dtype: str
    """Resolved torch dtype attribute name."""
