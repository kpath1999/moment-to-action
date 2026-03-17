"""Platform detection for the hardware abstraction layer.

Detects the current SoC/platform by reading the Qualcomm sysfs machine name
file.  This is more reliable than ``platform.machine()`` (which only tells us
the CPU architecture, not which specific SoC is present) and works correctly
inside containers and cross-compiled environments.

Usage::

    match detect_platform():
        case Platform.QCS6490:
            backend = QCS6490Backend(preferred_unit)
        case Platform.UNKNOWN:
            backend = QCS6490Backend(preferred_unit=ComputeUnit.CPU)
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger(__name__)

# Qualcomm sysfs file that contains the SoC/machine name (e.g. "QCS6490").
_QCOM_SOC_NAME_FILE = Path("/sys/devices/soc0/machine")


class Platform(Enum):
    """Known hardware platforms supported by this codebase."""

    QCS6490 = auto()
    """Qualcomm QCS6490 (Snapdragon 778G) — Hexagon HTP NPU, Adreno 642L GPU."""

    UNKNOWN = auto()
    """Unrecognised or development host — CPU inference only."""


def detect_platform() -> Platform:
    """Detect the current hardware platform via sysfs.

    Reads ``/sys/devices/soc0/machine`` (Qualcomm SOC name file) and maps
    the value to a :class:`Platform` enum member.  Falls back to
    ``Platform.UNKNOWN`` when the file is absent (dev machine, CI) or the
    SoC name is not recognised.

    Returns:
        The detected :class:`Platform`.
    """
    if not _QCOM_SOC_NAME_FILE.exists():
        logger.debug(
            "Sysfs file %s not found — assuming non-target host (Platform.UNKNOWN)",
            _QCOM_SOC_NAME_FILE,
        )
        return Platform.UNKNOWN

    soc_name = _QCOM_SOC_NAME_FILE.read_text().strip().upper()
    logger.debug("Detected SoC: %r", soc_name)

    if "QCS6490" in soc_name:
        return Platform.QCS6490

    logger.warning("Unrecognised SoC %r — falling back to Platform.UNKNOWN", soc_name)
    return Platform.UNKNOWN
