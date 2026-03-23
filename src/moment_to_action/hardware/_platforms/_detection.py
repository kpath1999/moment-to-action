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

import functools
import logging
import platform
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger(__name__)

# Qualcomm sysfs file that contains the SoC/machine name (e.g. "QCS6490").
_QCOM_SOC_NAME_FILE = Path("/sys/devices/soc0/machine")


class Platform(Enum):
    """Known hardware platforms supported by this codebase."""

    QCS6490 = auto()
    """Qualcomm QCS6490 (Snapdragon 778G) — Hexagon HTP NPU, Adreno 642L GPU."""

    X86_64 = auto()
    """Standard x86_64 laptop/desktop CPU (Intel/AMD)."""


@functools.cache
def detect_platform() -> Platform:
    """Detect the current hardware platform.

    Reads ``/sys/devices/soc0/machine`` if present and maps the SoC name to a
    :class:`Platform` member.  If the file is absent or the name is not
    recognised, raises ``RuntimeError`` with a consistent "Unrecognised SoC"
    message. This is only for Qualcomm devices.

    If that is not possible, the ``platform`` module is used as a fallback.

    Returns:
        The detected :class:`Platform`.

    Raises:
        RuntimeError: If the SoC cannot be identified.
    """
    #
    # Try qualcomm SoCs
    #
    soc_name = None

    if _QCOM_SOC_NAME_FILE.exists():
        # Read SoC name from sysfs
        soc_name = _QCOM_SOC_NAME_FILE.read_text().strip().upper()
        logger.debug("Detected SoC: %r", soc_name)

        if "QCS6490" in soc_name:
            return Platform.QCS6490

    #
    # Fallback: use platform module
    #
    machine = platform.machine().lower()
    logger.debug("CPU architecture: %r", machine)

    # Check for x86
    if machine in {"x86_64", "amd64"}:
        logger.info("Detected x86_64 architecture")
        return Platform.X86_64

    # Apple Silicon / generic ARM64 hosts still run CPU inference via the
    # generic backend (despite the historical X86_64 enum name).
    if machine in {"arm64", "aarch64"}:
        logger.info("Detected arm64 architecture; using CPU-only generic backend")
        return Platform.X86_64

    #
    # Not found/unknown
    #
    msg = (
        f"Unrecognised platform. SoC={soc_name!r}, arch={machine!r}. "
        "Add a new Platform member and backend to support this hardware."
    )
    raise RuntimeError(msg)
