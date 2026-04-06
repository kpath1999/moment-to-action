"""Power monitoring implementation for macOS arm64 (Apple Silicon).

Uses psutil-based CPU utilization estimates. Apple Silicon does not expose
per-core energy counters via a standard sysfs interface, so we always use
the heuristic estimator.

This is a dedicated macOS implementation so that future changes to the
x86_64 power monitor (e.g. RAPL-specific tuning) do not accidentally
break macOS.
"""

from __future__ import annotations

import logging
import time
from typing import ClassVar

import psutil

from moment_to_action.hardware._platforms._base import PowerMonitor
from moment_to_action.hardware._types import ComputeUnit, ComputeUnitUsageSample

logger = logging.getLogger(__name__)


class MacOSARM64PowerMonitor(PowerMonitor):
    """Estimate power on Apple Silicon via CPU utilization heuristics.

    Apple Silicon does not expose RAPL-style energy counters, so this
    monitor always uses a psutil-based estimate.  The heuristic is
    intentionally simple: base idle power + a load-proportional term
    scaled by CPU frequency.
    """

    _ESTIMATES: ClassVar[dict[ComputeUnit, float]] = {
        ComputeUnit.CPU: 50.0,  # Base estimate for Apple Silicon CPU
    }

    def sample(self, unit: ComputeUnit) -> ComputeUnitUsageSample:
        """Take a power measurement for *unit*.

        Args:
            unit: The compute unit to sample (macOS arm64 is CPU-only).

        Returns:
            A ``PowerSample`` with the estimated power reading.
        """
        if unit != ComputeUnit.CPU:
            return ComputeUnitUsageSample(
                timestamp=time.time(),
                device=unit,
                power_mw=0.0,
                usage_pct=0.0,
            )

        return self._estimate()

    def _estimate(self) -> ComputeUnitUsageSample:
        """Estimate power using CPU frequency and utilization heuristics.

        Rough estimate: base_power + (freq_ghz x util_pct x factor).
        """
        cpu_util = psutil.cpu_percent(interval=None)

        try:
            freq_info = psutil.cpu_freq()
            freq_ghz = freq_info.current / 1000.0 if freq_info else 3.0
        except (AttributeError, OSError):
            freq_ghz = 3.0  # Fallback: assume ~3 GHz for Apple Silicon

        base_power = self._ESTIMATES[ComputeUnit.CPU]
        load_power = freq_ghz * cpu_util * 0.6

        return ComputeUnitUsageSample(
            timestamp=time.time(),
            device=ComputeUnit.CPU,
            power_mw=base_power + load_power,
            usage_pct=cpu_util,
        )
