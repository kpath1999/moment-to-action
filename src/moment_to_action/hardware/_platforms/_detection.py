"""Platform detection for the hardware abstraction layer.

Detects the current SoC/platform by reading the Qualcomm sysfs machine name
file.  This is more reliable than ``platform.machine()`` (which only tells us
the CPU architecture, not which specific SoC is present) and works correctly
inside containers and cross-compiled environments.

Usage::

    platform = detect_platform()  # raises RuntimeError if unrecognised
    backend = QCS6490Backend(preferred_unit)
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


def detect_platform() -> Platform:
    """Detect the current hardware platform via sysfs.

    Reads ``/sys/devices/soc0/machine`` (Qualcomm SOC name file) and maps
    the value to a :class:`Platform` enum member.  If the file is absent the
    host is not a Qualcomm device; if the SoC name is unrecognised the device
    is not yet supported — both are hard errors.

    Returns:
        The detected :class:`Platform`.

    Raises:
        RuntimeError: If the host is not a supported Qualcomm device.
    """
    try:
        soc_name = _QCOM_SOC_NAME_FILE.read_text().strip().upper()
    except FileNotFoundError:
        # sysfs file absent → definitely not a Qualcomm SoC.
        msg = "Unsupported platform: not running on a Qualcomm device (sysfs machine file absent)."
        raise RuntimeError(msg) from None

    logger.debug("Detected SoC: %r", soc_name)

    if "QCS6490" in soc_name:
        return Platform.QCS6490

    msg = (
        f"Unsupported Qualcomm SoC {soc_name!r}. "
        "Add a new Platform member and backend to support this hardware."
    )
    raise RuntimeError(msg)
