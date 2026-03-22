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
    """A single power measurement snapshot for one compute unit."""

    timestamp: float
    """Unix timestamp of the measurement (seconds)."""

    compute_unit: ComputeUnit
    """The unit that was active during sampling."""

    power_mw: float
    """Power draw in milliwatts."""

    utilization_pct: float
    """Utilisation percentage (0-100)."""
