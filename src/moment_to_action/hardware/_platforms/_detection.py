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
    the value to a :class:`Platform` enum member.

    Returns:
        The detected :class:`Platform`.

    Raises:
        RuntimeError: If the sysfs file is absent or the SoC name is not
            recognised.  This is intentional — callers must handle unsupported
            hardware explicitly rather than silently running on a wrong platform.
    """
    if not _QCOM_SOC_NAME_FILE.exists():
        msg = (
            f"Platform detection failed: {_QCOM_SOC_NAME_FILE} not found. "
            "Are you running on supported hardware?"
        )
        raise RuntimeError(msg)

    soc_name = _QCOM_SOC_NAME_FILE.read_text().strip().upper()
    logger.debug("Detected SoC: %r", soc_name)

    if "QCS6490" in soc_name:
        return Platform.QCS6490

    msg = (
        f"Unrecognised SoC {soc_name!r}. "
        "Add a new Platform member and backend to support this hardware."
    )
    raise RuntimeError(msg)
