"""Hardware type definitions — enums and data models.

Kept separate to avoid circular imports: stages and preprocessors
import ``ComputeUnit`` from here, not from the backend module.
"""

from __future__ import annotations

from enum import Enum, auto

from pydantic import BaseModel


class ComputeUnit(Enum):
    """Available compute units on a hardware accelerator platform."""

    CPU = auto()
    NPU = auto()
    GPU = auto()
    DSP = auto()


class PowerSample(BaseModel):
    """A single power measurement snapshot for one compute unit.

    Attributes:
        timestamp: Unix timestamp of the measurement (seconds).
        compute_unit: The unit that was active during sampling.
        power_mw: Power draw in milliwatts.
        utilization_pct: Utilisation percentage (0-100).
    """

    timestamp: float
    compute_unit: ComputeUnit
    power_mw: float
    utilization_pct: float
