"""Power monitoring implementation for the QCS6490 platform.

Reads power draw from sysfs when hardware sensors are available; otherwise
returns static estimates derived from typical QCS6490 power envelopes.

Utilization is read via:
- **CPU**: ``psutil.cpu_percent()`` — cross-platform, accurate
- **GPU**: ``/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage`` — Qualcomm Adreno sysfs
- **NPU/DSP**: not available via a stable public sysfs interface; reported as 0.0
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import ClassVar

import psutil

from moment_to_action.hardware._platforms._base import PowerMonitor
from moment_to_action.hardware._types import ComputeUnit, PowerSample

logger = logging.getLogger(__name__)

# Adreno GPU utilization sysfs path (Qualcomm kgsl driver).
_KGSL_GPU_BUSY_PATH = Path("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage")


class QCS6490PowerMonitor(PowerMonitor):
    """Reads power from sysfs on the QCS6490; falls back to estimates.

    Power estimates are based on typical operating power envelopes for the
    Snapdragon 778G SoC (QCS6490):
      - NPU (Hexagon HTP): ~500 mW sustained inference load
      - GPU (Adreno 642L): ~800 mW sustained load
      - DSP (CDSP):        ~150 mW
      - CPU (Kryo 670):    ~300 mW multi-core load
    These are approximate mid-load figures, not TDP or peak values.
    """

    SYSFS_POWER_PATH: ClassVar[str] = "/sys/class/power_supply"

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
        # NOTE: the battery power_now file reports *total* system power draw,
        # not per-unit.  We pass `unit` through so the caller knows which
        # unit was active when the sample was taken.
        try:
            power_uw = int(Path(f"{self.SYSFS_POWER_PATH}/battery/power_now").read_text().strip())
            return PowerSample(
                timestamp=time.time(),
                compute_unit=unit,
                power_mw=power_uw / 1000.0,
                utilization_pct=self._read_utilization(unit),
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning("HW power sensor read failed: %s", e)
            return self._estimate(unit)

    def _estimate(self, unit: ComputeUnit) -> PowerSample:
        return PowerSample(
            timestamp=time.time(),
            compute_unit=unit,
            power_mw=self._ESTIMATES.get(unit, 300.0),
            utilization_pct=self._read_utilization(unit),
        )

    @staticmethod
    def _read_utilization(unit: ComputeUnit) -> float:
        """Return utilization percentage (0-100) for *unit*.

        - CPU: ``psutil.cpu_percent()`` (instantaneous, non-blocking)
        - GPU: Adreno kgsl sysfs ``gpu_busy_percentage``
        - NPU/DSP: no stable public sysfs interface available; returns 0.0
        """
        if unit == ComputeUnit.CPU:
            # interval=None returns the value since the last call (non-blocking).
            return psutil.cpu_percent(interval=None)

        if unit == ComputeUnit.GPU and _KGSL_GPU_BUSY_PATH.exists():
            try:
                return float(_KGSL_GPU_BUSY_PATH.read_text().strip())
            except (ValueError, OSError) as e:
                logger.debug("GPU busy read failed: %s", e)

        # NPU and DSP utilization is not available via a stable public interface.
        return 0.0
