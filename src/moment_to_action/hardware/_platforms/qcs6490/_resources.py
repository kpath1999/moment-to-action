"""Resource monitoring implementation for the QCS6490 platform.

Power readings are disabled — the battery sysfs sensor reports whole-system
draw and is unreliable for per-unit attribution.  ``power_mw`` is always 0.0.

Utilization and frequency are read via:
- **CPU**: ``psutil.cpu_percent()`` / ``psutil.cpu_freq()`` — cross-platform
- **GPU**: ``/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage`` — Adreno busy %
           ``/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq``    — Adreno clock (Hz)
- **NPU/DSP**: not available via a stable public sysfs interface; reported as 0.0
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import psutil

from moment_to_action.hardware._resource_monitor import ResourceMonitor
from moment_to_action.hardware._types import ComputeUnit, ComputeUnitUsageSample

logger = logging.getLogger(__name__)


# Adreno GPU sysfs paths (Qualcomm kgsl driver).
_KGSL_GPU_BUSY_PATH = Path("/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage")
_KGSL_GPU_FREQ_PATH = Path("/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq")


class QCS6490ResourceMonitor(ResourceMonitor):
    """Resource monitor for the QCS6490 (Snapdragon 778G / Adreno 642L).

    Power is always reported as 0.0 — the battery sysfs node exposes total
    system draw, not per-unit, so it is not meaningful here.
    Utilization and frequency are read from kgsl sysfs where available.
    """

    def sample(self, unit: ComputeUnit) -> ComputeUnitUsageSample:
        """Take a resource measurement for *unit*.

        Args:
            unit: The compute unit to sample.

        Returns:
            A ``ComputeUnitUsageSample`` with utilization, frequency, and
            memory readings.  ``power_mw`` is always 0.0.
        """
        return ComputeUnitUsageSample(
            timestamp=datetime.now(tz=timezone.utc),
            device=unit,
            usage_pct=self._read_utilization(unit),
            frequency_mhz=self._read_frequency_mhz(unit),
            memory_mb=self.used_memory_mb(),
            power_mw=0.0,
        )

    @staticmethod
    def _read_frequency_mhz(unit: ComputeUnit) -> float:
        """Return operating frequency in MHz for *unit*, or 0.0 if unavailable.

        Args:
            unit: The compute unit whose frequency to query.

        Returns:
            Frequency in MHz, or 0.0 when the sysfs/psutil source is absent.
        """
        if unit == ComputeUnit.CPU:
            try:
                freq_info = psutil.cpu_freq()
            except (AttributeError, OSError):
                return 0.0
            else:
                return freq_info.current if freq_info else 0.0

        if unit == ComputeUnit.GPU and _KGSL_GPU_FREQ_PATH.exists():
            try:
                return float(_KGSL_GPU_FREQ_PATH.read_text().strip()) / 1_000_000
            except (ValueError, OSError) as e:
                logger.debug("GPU freq read failed: %s", e)

        return 0.0

    @staticmethod
    def _read_utilization(unit: ComputeUnit) -> float:
        """Return utilization percentage (0-100) for *unit*.

        - CPU: ``psutil.cpu_percent()`` (instantaneous, non-blocking)
        - GPU: Adreno kgsl sysfs ``gpu_busy_percentage``
        - NPU/DSP: no stable public sysfs interface available; returns 0.0

        Args:
            unit: The compute unit to query.

        Returns:
            Utilization in percent, or 0.0 when unavailable.
        """
        if unit == ComputeUnit.CPU:
            return psutil.cpu_percent(interval=None)

        if unit == ComputeUnit.GPU and _KGSL_GPU_BUSY_PATH.exists():
            try:
                return float(_KGSL_GPU_BUSY_PATH.read_text().strip())
            except (ValueError, OSError) as e:
                logger.debug("GPU busy read failed: %s", e)

        return 0.0
