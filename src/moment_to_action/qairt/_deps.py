"""System dependency checking for QAIRT SDK installation."""

from __future__ import annotations

import logging
import shutil
import subprocess

_LOG = logging.getLogger(__name__)

_APT_DEPS: list[str] = [
    "libncurses5",
    "libgl1",
    "clang",
    "libc++-dev",
    "libc++abi-dev",
    "flatbuffers-compiler",
    "libflatbuffers-dev",
    "rename",
    "unzip",
    "zip",
    "ca-certificates",
    "curl",
    "locales",
    "lsb-release",
    "wget",
]

# Fedora/RHEL equivalents.
# Notes:
# - `rename` (Perl rename) → `prename` on Fedora 35+.
# - `lsb-release` → `redhat-lsb` on Fedora 35+ (`redhat-lsb-core` removed in Fedora 34).
_DNF_DEPS: list[str] = [
    "ncurses-compat-libs",
    "mesa-libGL",
    "clang",
    "libcxx-devel",
    "libcxxabi-devel",
    "flatbuffers",
    "flatbuffers-devel",
    "prename",
    "unzip",
    "zip",
    "ca-certificates",
    "curl",
    "glibc-langpack-en",
    "redhat-lsb",
    "wget2",
]


def _missing_apt(deps: list[str]) -> list[str]:
    missing: list[str] = []
    for p in deps:
        cmd = ["dpkg", "-s", p]
        if subprocess.run(cmd, capture_output=True, check=False).returncode != 0:  # noqa: S603
            missing.append(p)
    return missing


def _missing_dnf(deps: list[str]) -> list[str]:
    missing: list[str] = []
    for p in deps:
        cmd = ["rpm", "-q", p]
        if subprocess.run(cmd, capture_output=True, check=False).returncode != 0:  # noqa: S603
            missing.append(p)
    return missing


def check_system_deps() -> list[str]:
    """Return list of missing system package names for the current distro.

    On non-Ubuntu systems a warning is logged; package checks still run where
    possible but results may differ from the officially supported platform.
    """
    if shutil.which("dpkg"):
        return _missing_apt(_APT_DEPS)
    if shutil.which("rpm"):
        _LOG.warning(
            "QAIRT SDK is officially supported on Ubuntu only; "
            "package check results may differ on this system."
        )
        return _missing_dnf(_DNF_DEPS)
    _LOG.warning(
        "Cannot check system dependencies: no dpkg or rpm found. "
        "QAIRT SDK is officially supported on Ubuntu only."
    )
    return []
