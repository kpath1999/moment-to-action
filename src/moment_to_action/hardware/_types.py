"""Hardware type definitions — enums and data models.

Kept separate to avoid circular imports: stages and preprocessors
import ``ComputeUnit`` from here, not from the backend module.
"""

from __future__ import annotations

import typing as t
from enum import Enum
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from datetime import datetime

    import torch


class ComputeUnit(str, Enum):
    """Available compute units on a hardware accelerator platform."""

    CPU = "CPU"
    GPU = "GPU"
    NPU = "NPU"
    DSP = "DSP"


@attrs.frozen
class ComputeUnitUsageSample:
    """A single compute unit usage measurement snapshot for one compute unit."""

    timestamp: datetime
    """Time when the sample was taken."""

    device: ComputeUnit
    """The device that was active during sampling."""

    usage_pct: float
    """Utilisation percentage (0-100)."""

    frequency_mhz: float
    """Operating frequency in MHz."""

    memory_mb: float
    """Memory usage in megabytes (across ALL processes)."""

    power_mw: float
    """Power draw in milliwatts."""

    def json(self) -> dict[str, t.Any]:
        """Generate a JSON-serializable dictionary representation of this sample."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "device": self.device,
            "usage_pct": self.usage_pct,
            "frequency_mhz": self.frequency_mhz,
            "memory_mb": self.memory_mb,
            "power_mw": self.power_mw,
        }


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
