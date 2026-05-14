"""Unit tests for qairt._deps system dependency checking."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from moment_to_action.qairt._deps import (
    _APT_DEPS,
    _DNF_DEPS,
    _missing_apt,
    _missing_dnf,
    check_system_deps,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_run(missing: set[str]) -> Callable[..., object]:
    """Return a subprocess.run side-effect that reports packages in missing as absent."""

    def _run(cmd: list[str], **_: object) -> object:
        pkg = cmd[-1]

        class _R:
            returncode = 1 if pkg in missing else 0

        return _R()

    return _run


@pytest.mark.unit
class TestMissingApt:
    """Tests for _missing_apt."""

    def test_all_present_returns_empty(self) -> None:
        """Returns empty list when all packages are installed."""
        with patch("moment_to_action.qairt._deps.subprocess.run", side_effect=_make_run(set())):
            assert _missing_apt(_APT_DEPS) == []

    def test_missing_packages_returned(self) -> None:
        """Returns list of packages that are not installed."""
        missing = {"clang", "unzip"}
        with patch("moment_to_action.qairt._deps.subprocess.run", side_effect=_make_run(missing)):
            result = _missing_apt(_APT_DEPS)
        assert set(result) == missing

    def test_uses_dpkg_s(self) -> None:
        """Checks package presence using dpkg -s."""
        calls: list[list[str]] = []

        def _run(cmd: list[str], **_: object) -> object:
            calls.append(cmd)

            class _R:
                returncode = 0

            return _R()

        with patch("moment_to_action.qairt._deps.subprocess.run", side_effect=_run):
            _missing_apt(["clang"])

        assert calls[0][:2] == ["dpkg", "-s"]


@pytest.mark.unit
class TestMissingDnf:
    """Tests for _missing_dnf."""

    def test_all_present_returns_empty(self) -> None:
        """Returns empty list when all packages are installed."""
        with patch("moment_to_action.qairt._deps.subprocess.run", side_effect=_make_run(set())):
            assert _missing_dnf(_DNF_DEPS) == []

    def test_missing_packages_returned(self) -> None:
        """Returns list of packages that are not installed."""
        missing = {"clang", "wget2"}
        with patch("moment_to_action.qairt._deps.subprocess.run", side_effect=_make_run(missing)):
            result = _missing_dnf(_DNF_DEPS)
        assert set(result) == missing

    def test_uses_rpm_q(self) -> None:
        """Checks package presence using rpm -q."""
        calls: list[list[str]] = []

        def _run(cmd: list[str], **_: object) -> object:
            calls.append(cmd)

            class _R:
                returncode = 0

            return _R()

        with patch("moment_to_action.qairt._deps.subprocess.run", side_effect=_run):
            _missing_dnf(["clang"])

        assert calls[0][:2] == ["rpm", "-q"]


@pytest.mark.unit
class TestCheckSystemDeps:
    """Tests for check_system_deps."""

    def test_apt_path_used_when_dpkg_present(self) -> None:
        """Uses apt checking when dpkg is available."""
        with patch("moment_to_action.qairt._deps.shutil.which", return_value="/usr/bin/dpkg"):
            with patch(
                "moment_to_action.qairt._deps._missing_apt", return_value=["libgl1"]
            ) as mock_apt:
                result = check_system_deps()
        mock_apt.assert_called_once()
        assert result == ["libgl1"]

    def test_dnf_path_used_when_only_rpm_present(self) -> None:
        """Uses rpm checking when dpkg is absent but rpm is available."""

        def _which(name: str) -> str | None:
            return "/usr/bin/rpm" if name == "rpm" else None

        with patch("moment_to_action.qairt._deps.shutil.which", side_effect=_which):
            with patch("moment_to_action.qairt._deps._missing_dnf", return_value=[]) as mock_dnf:
                result = check_system_deps()
        mock_dnf.assert_called_once()
        assert result == []

    def test_unknown_distro_returns_empty_and_warns(self) -> None:
        """Returns empty list and logs warning when neither dpkg nor rpm is available."""
        with patch("moment_to_action.qairt._deps.shutil.which", return_value=None):
            with patch("moment_to_action.qairt._deps._LOG") as mock_log:
                result = check_system_deps()
        assert result == []
        mock_log.warning.assert_called_once()

    def test_rpm_distro_warns_about_support(self) -> None:
        """Logs a support warning on rpm-based systems."""

        def _which(name: str) -> str | None:
            return "/usr/bin/rpm" if name == "rpm" else None

        with patch("moment_to_action.qairt._deps.shutil.which", side_effect=_which):
            with patch("moment_to_action.qairt._deps._missing_dnf", return_value=[]):
                with patch("moment_to_action.qairt._deps._LOG") as mock_log:
                    check_system_deps()
        mock_log.warning.assert_called_once()
