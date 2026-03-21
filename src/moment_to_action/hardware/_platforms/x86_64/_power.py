"""Power monitoring implementation for the x86_64 platform.

Reads power draw from Intel RAPL sysfs when available; otherwise estimates
based on CPU frequency and utilization.

**Intel RAPL (Running Average Power Limit):**
  - Available on most Intel/AMD CPUs since ~2012
  - Reads from ``/sys/class/powercap/intel-rapl:0/energy_uj``
  - Energy is cumulative; power is calculated as dE / dt
  - Typical CPU TDP: 65-125 W (laptops), 35-65 W (ultrabooks)

**Fallback (psutil-based):**
  - When RAPL is unavailable, estimate using CPU frequency x utilization
  - Rough heuristic: power ~= freq (GHz) x util (%) x base_tdp_estimate
  - Base estimate for x86_64: ~60 mW per 1% at base frequency
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

# Intel RAPL sysfs paths — check for availability at runtime.
_RAPL_BASE_PATH = Path("/sys/class/powercap/intel-rapl:0")
_RAPL_ENERGY_PATH = _RAPL_BASE_PATH / "energy_uj"


class X86_64PowerMonitor(PowerMonitor):  # noqa: N801
    """Reads power from Intel RAPL sysfs on x86_64; falls back to estimates.

    Power estimates are based on rough heuristics for typical x86_64 CPUs:
      - CPU: ~50-80 mW base (idle) + variable per-core load
      - Estimate: CPU frequency (GHz) x utilization (%) x ~6 mW per GHz
      - Typical laptop TDP: 65-125 W (includes other components)
    """

    _ESTIMATES: ClassVar[dict[ComputeUnit, float]] = {
        ComputeUnit.CPU: 50.0,  # Base estimate for x86_64 CPU
    }

    def __init__(self) -> None:
        """Initialize RAPL sensor availability."""
        self._rapl_available = _RAPL_ENERGY_PATH.exists()
        self._last_energy_uj: float | None = None
        self._last_time: float | None = None

        if self._rapl_available:
            logger.info("Intel RAPL available at %s", _RAPL_ENERGY_PATH)
        else:
            logger.debug("Intel RAPL not available — using psutil estimates")

    def sample(self, unit: ComputeUnit) -> PowerSample:
        """Take a power measurement for *unit*.

        Reads from Intel RAPL when available; falls back to psutil-based
        estimates.

        Args:
            unit: The compute unit to sample (x86_64 is CPU-only).

        Returns:
            A ``PowerSample`` with the current power reading.
        """
        if unit != ComputeUnit.CPU:
            # x86_64 is CPU-only; other units return zero.
            return PowerSample(
                timestamp=time.time(),
                compute_unit=unit,
                power_mw=0.0,
                utilization_pct=0.0,
            )

        if self._rapl_available:
            return self._read_rapl()
        return self._estimate()

    def _read_rapl(self) -> PowerSample:
        """Read power from Intel RAPL energy counter.

        RAPL reports cumulative energy in microjoules. Compute power as
        dE / dt in watts.

        Returns:
            A PowerSample with RAPL-based power reading.
        """
        try:
            energy_uj = int(_RAPL_ENERGY_PATH.read_text().strip())
            now = time.time()

            power_mw = 0.0
            if self._last_energy_uj is not None and self._last_time is not None:
                # energy_uj is in microjoules; convert to milliwatts
                # power = ΔE (μJ) / Δt (s) = ΔE (μJ) / (Δt * 10^6) = ΔE (mJ) / Δt
                delta_energy_uj = energy_uj - self._last_energy_uj
                delta_time_s = now - self._last_time

                if delta_time_s > 0:
                    power_mw = (delta_energy_uj / 1000.0) / delta_time_s

            self._last_energy_uj = energy_uj
            self._last_time = now

            return PowerSample(
                timestamp=now,
                compute_unit=ComputeUnit.CPU,
                power_mw=power_mw,
                utilization_pct=psutil.cpu_percent(interval=None),
            )
        except (FileNotFoundError, ValueError, OSError) as e:
            logger.warning("RAPL read failed: %s", e)
            return self._estimate()

    def _estimate(self) -> PowerSample:
        """Estimate power using CPU frequency and utilization heuristics.

        Rough estimate: base_power + (freq_ghz x util_pct x factor).
        """
        cpu_util = psutil.cpu_percent(interval=None)

        # Get CPU frequency in GHz (average across cores, or base frequency)
        try:
            freq_info = psutil.cpu_freq()
            freq_ghz = freq_info.current / 1000.0 if freq_info else 2.0
        except (AttributeError, OSError):
            freq_ghz = 2.0  # Fallback: assume ~2 GHz

        # Rough heuristic: 60 mW per GHz at 100% utilization, baseline 50 mW
        base_power = self._ESTIMATES[ComputeUnit.CPU]
        load_power = freq_ghz * cpu_util * 0.6  # 0.6 mW per GHz per 1% utilization

        return PowerSample(
            timestamp=time.time(),
            compute_unit=ComputeUnit.CPU,
            power_mw=base_power + load_power,
            utilization_pct=cpu_util,
        )
