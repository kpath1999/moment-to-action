"""Power monitoring implementation for the QCS6490 platform.

Reads power draw from sysfs when hardware sensors are available; otherwise
returns static estimates derived from typical QCS6490 power envelopes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import ClassVar

from moment_to_action.hardware._platforms._base import PowerMonitor
from moment_to_action.hardware._types import ComputeUnit, PowerSample

logger = logging.getLogger(__name__)


class QCS6490PowerMonitor(PowerMonitor):
    """Reads power from sysfs on the QCS6490; falls back to estimates.

    Estimates are based on typical power envelopes for each compute unit
    and are used when hardware sensors are unavailable (e.g., in CI/dev).
    """

    SYSFS_POWER_PATH: ClassVar[str] = "/sys/class/power_supply"

    # Static milliwatt estimates when sysfs sensors are absent.
    _ESTIMATES: ClassVar[dict[ComputeUnit, float]] = {
        ComputeUnit.NPU: 500.0,
        ComputeUnit.GPU: 800.0,
        ComputeUnit.DSP: 150.0,
        ComputeUnit.CPU: 300.0,
    }

    def __init__(self) -> None:
        self._hw_available = Path(self.SYSFS_POWER_PATH).exists()

    def sample(self, unit: ComputeUnit) -> PowerSample:
        """Take a power measurement for *unit*.

        Reads from sysfs when available; falls back to static estimates.

        Args:
            unit: The compute unit to sample.

        Returns:
            A ``PowerSample`` with the current power reading.
        """
        if self._hw_available:
            return self._read_hw_sensor(unit)
        return self._estimate(unit)

    def _read_hw_sensor(self, unit: ComputeUnit) -> PowerSample:
        try:
            with open(f"{self.SYSFS_POWER_PATH}/battery/power_now") as f:  # noqa: PTH123
                power_uw = int(f.read().strip())
            return PowerSample(
                timestamp=time.time(),
                compute_unit=unit,
                power_mw=power_uw / 1000.0,
                utilization_pct=0.0,
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning("HW power sensor read failed: %s", e)
            return self._estimate(unit)

    def _estimate(self, unit: ComputeUnit) -> PowerSample:
        return PowerSample(
            timestamp=time.time(),
            compute_unit=unit,
            power_mw=self._ESTIMATES.get(unit, 300.0),
            utilization_pct=0.0,
        )
