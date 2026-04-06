"""Hardware type definitions — enums and data models.

Kept separate to avoid circular imports: stages and preprocessors
import ``ComputeUnit`` from here, not from the backend module.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    import torch


class ComputeUnit(StrEnum):
    """Available compute units on a hardware accelerator platform."""

    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"
    DSP = "DSP"


@attrs.frozen
class ComputeUnitUsageSample:
    """A single compute unit usage measurement snapshot for one compute unit."""

    timestamp: float
    """Unix timestamp of the measurement (seconds)."""

    device: ComputeUnit
    """The device that was active during sampling."""

    usage_pct: float
    """Utilisation percentage (0-100)."""

    power_mw: float
    """Power draw in milliwatts."""


@attrs.frozen
class TorchExecutionPolicy:
    """Resolved torch execution configuration for model loading and inference.

    Attributes:
        device: Resolved ``torch.device`` instance.
        dtype: Resolved ``torch.dtype`` instance.
    """

    device: torch.device
    """Resolved torch device."""

    dtype: torch.dtype
    """Resolved torch dtype."""
