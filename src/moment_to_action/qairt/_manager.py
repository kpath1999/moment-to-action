"""QAIRT SDK manager."""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import TYPE_CHECKING

from moment_to_action.qairt._deps import check_system_deps as _check_system_deps

if TYPE_CHECKING:
    import numpy as np

    from moment_to_action.config import AppConfig
    from moment_to_action.paths import PathManager

_log = logging.getLogger(__name__)

_QAIRT_VM = Path(sys.executable).parent / "qairt-vm"

_ERR_NOT_INSTALLED = "SDK not installed; run 'm2a qairt install' first"
_ERR_NOT_INSTALLED_M2A = "SDK not installed via m2a"
_ERR_FETCH_PATH = "fetch succeeded but SDK path not found under install dir"
_ERR_ALREADY_INSTALLED = "QAIRT SDK {version} already installed at {path}"


class QairtSDKManager:
    """Manages the QAIRT SDK installation for this application."""

    def __init__(self, sdk_path: Path | None, sdk_version: str, install_dir: Path) -> None:
        """Initialize the QAIRT SDK manager.

        Args:
            sdk_path: Path to the installed SDK, or None if not yet installed.
            sdk_version: Version string to fetch (e.g. "2.45.0").
            install_dir: Directory where SDK will be installed.
        """
        self._sdk_path = sdk_path
        self._sdk_version = sdk_version
        self._install_dir = install_dir

        _log.debug(
            f"Initialized QairtSDKManager: version={sdk_version}, "
            f"install_dir={install_dir}, sdk_path={sdk_path}"
        )

    @classmethod
    def from_app_config(cls, config: AppConfig, path_manager: PathManager) -> QairtSDKManager:
        """Construct from application config and path manager."""
        return cls(
            sdk_path=config.qairt_sdk_path,
            sdk_version=config.qairt_sdk_version,
            install_dir=path_manager.data.data_dir,
        )

    # --- Properties (pure state, no subprocess) ---

    @property
    def is_available(self) -> bool:
        """True if SDK path is configured and the directory exists on disk."""
        return self._sdk_path is not None and self._sdk_path.exists()

    @property
    def path(self) -> Path | None:
        """Configured SDK path, or None if not installed."""
        return self._sdk_path

    @property
    def configured_version(self) -> str:
        """Version string that will be fetched on install."""
        return self._sdk_version

    @property
    def installed_version(self) -> str | None:
        """Full version string (e.g. '2.45.0.24') from the installed directory name.

        Returns None if SDK is not available.
        """
        if not self.is_available:
            return None
        assert self._sdk_path is not None  # noqa: S101
        return self._sdk_path.name

    # --- Environment ---

    def configure_env(self) -> None:
        """Set QAIRT_SDK_ROOT, ADSP_LIBRARY_PATH, and pre-load libpython.

        QAIRT's native pybind extension (libPyNetRun) lists libpython3.10.so.1.0 as a
        NEEDED dependency. When Python is managed by uv its interpreter is a static
        build, so libpython is not already loaded in the process. The dynamic linker
        also won't find it via LD_LIBRARY_PATH changes made after process start. We
        therefore pre-load libpython via its absolute sysconfig path with RTLD_GLOBAL;
        this registers it in the loaded-libs table under its SONAME, satisfying
        libPyNetRun's dependency check without any filesystem search.

        ADSP_LIBRARY_PATH is prepended with the SDK's ``lib/hexagon-v*/unsigned``
        directories so that the DSP skel libraries bundled with this SDK version are
        found before any system-installed skels. A version mismatch between the HTP
        stub (in the SDK) and the skel (on the device) causes the FastRPC transport
        CRC check to fail; keeping them in sync prevents that.

        Raises:
            RuntimeError: If SDK path is not configured.
        """
        if self._sdk_path is None:
            _log.error("Cannot configure environment: SDK path not set")
            raise RuntimeError(_ERR_NOT_INSTALLED)
        _log.info(f"Setting QAIRT_SDK_ROOT={self._sdk_path}")
        os.environ["QAIRT_SDK_ROOT"] = str(self._sdk_path)

        # Prepend versioned hexagon skel dirs so they shadow any system-installed skels
        hexagon_dirs = sorted((self._sdk_path / "lib").glob("hexagon-v*/unsigned"))
        if hexagon_dirs:
            adsp_paths = ":".join(str(p) for p in hexagon_dirs)
            existing = os.environ.get("ADSP_LIBRARY_PATH", "")
            os.environ["ADSP_LIBRARY_PATH"] = f"{adsp_paths}:{existing}" if existing else adsp_paths
            _log.info(f"Setting ADSP_LIBRARY_PATH={os.environ['ADSP_LIBRARY_PATH']}")

        lib_dir = sysconfig.get_config_var("LIBDIR")
        lib_name = sysconfig.get_config_var("INSTSONAME")
        if lib_dir and lib_name:
            lib_path = Path(lib_dir) / lib_name
            if lib_path.exists():
                _log.debug(f"Pre-loading {lib_path} with RTLD_GLOBAL for QAIRT native extensions")
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

    # --- Operations ---

    def check_system_deps(self) -> list[str]:
        """Return list of missing system package names for the current distro."""
        missing = _check_system_deps()
        if missing:
            _log.warning(f"Missing system dependencies: {', '.join(missing)}")
        else:
            _log.debug("All system dependencies present")
        return missing

    def install(self, *, stream: bool = True) -> Path:
        """Fetch the SDK and return the extracted path.

        Args:
            stream: If True, let qairt-vm output flow to the terminal. If False,
                capture output (useful for --json mode).

        Raises:
            RuntimeError: If already installed, fetch fails, or extracted path cannot be found.
        """
        if self.is_available:
            msg = _ERR_ALREADY_INSTALLED.format(version=self.installed_version, path=self._sdk_path)
            _log.error(msg)
            raise RuntimeError(msg)
        _log.info(
            f"Starting QAIRT SDK install: version={self._sdk_version}, "
            f"install_dir={self._install_dir}"
        )
        result = subprocess.run(  # noqa: S603
            [_QAIRT_VM, "fetch", "--version", self._sdk_version, "--dir", str(self._install_dir)],
            capture_output=not stream,
            check=False,
        )
        if result.returncode != 0:
            msg = f"qairt-vm fetch failed (exit {result.returncode})"
            _log.error(msg)
            raise RuntimeError(msg)
        _log.debug("qairt-vm fetch succeeded, locating SDK path")
        path = self._find_sdk_path()
        if path is None:
            _log.error(_ERR_FETCH_PATH)
            raise RuntimeError(_ERR_FETCH_PATH)
        self._sdk_path = path
        self._cleanup_zip_files()
        _log.info(f"Successfully installed QAIRT SDK {self._sdk_version} at {path}")
        return path

    def verify(self, *, stream: bool = True) -> list[str]:
        """Check SDK installation and return any warnings.

        On Ubuntu (dpkg available) runs ``qairt-vm --inspect`` for full
        verification. On other systems logs a support warning and checks path
        and system deps only.

        Args:
            stream: If True, let qairt-vm output flow to the terminal (Ubuntu only).

        Returns:
            List of warning strings for any issues found. Empty means all good.

        Raises:
            RuntimeError: If SDK path is not configured.
        """
        if self._sdk_path is None:
            _log.error("Cannot verify: SDK path not set")
            raise RuntimeError(_ERR_NOT_INSTALLED)
        _log.info(f"Starting QAIRT SDK verification at {self._sdk_path}")
        issues: list[str] = []
        if not self._sdk_path.exists():
            issue = f"SDK path does not exist: {self._sdk_path}"
            issues.append(issue)
            _log.error(issue)
        else:
            _log.debug(f"SDK path exists: {self._sdk_path}")
        missing_deps = self.check_system_deps()
        issues.extend(f"Missing system package: {pkg}" for pkg in missing_deps)
        if shutil.which("dpkg"):
            _log.debug("Running qairt-vm --inspect on Ubuntu")
            env = {**os.environ, "QAIRT_SDK_ROOT": str(self._sdk_path)}
            result = subprocess.run(  # noqa: S603
                [_QAIRT_VM, "--inspect"],
                capture_output=not stream,
                text=True,
                env=env,
                check=False,
            )
            if result.returncode != 0:
                issue = f"qairt-vm --inspect failed (exit {result.returncode})"
                issues.append(issue)
                _log.warning(issue)
            else:
                _log.debug("qairt-vm --inspect passed")
        else:
            issue = (
                "QAIRT SDK is officially supported on Ubuntu only; "
                "--inspect skipped on this system."
            )
            issues.append(issue)
            _log.warning(issue)
        if issues:
            _log.warning(f"Verification found {len(issues)} issue(s)")
        else:
            _log.info("Verification passed: no issues found")
        return issues

    def convert(
        self,
        input_path: Path,
        output_path: Path,
        calibration_data: np.ndarray,
        *,
        stream: bool = True,  # noqa: ARG002
    ) -> Path:
        """Convert an ONNX model to quantized DLC using the QAIRT Python API.

        Builds a CalibrationConfig from ``calibration_data`` and calls
        ``qairt.convert`` to produce an INT8-quantized ``qairt.Model``, then
        saves it via ``model.save``.  Does NOT call ``qairt.compile`` —
        that produces a device-specific ``.bin``, not a portable ``.dlc``.

        Args:
            input_path: Path to the source ONNX model.
            output_path: Destination path for the ``.dlc`` output file.
            calibration_data: Float32 array of shape ``(N, C, H, W)`` —
                stacked preprocessed calibration images used for INT8
                quantization.
            stream: Unused; kept for API symmetry with ``install`` and
                ``verify``.

        Returns:
            Resolved output path.

        Raises:
            RuntimeError: If the SDK is not available or conversion fails.
        """
        if not self.is_available:
            raise RuntimeError(_ERR_NOT_INSTALLED)
        import qairt  # noqa: PLC0415

        try:
            calib_config = qairt.CalibrationConfig(dataset=calibration_data)
            dlc = qairt.convert(str(input_path), calibration_config=calib_config)
            dlc.save(str(output_path))
        except Exception as exc:
            msg = f"QAIRT conversion failed: {exc}"
            raise RuntimeError(msg) from exc
        return output_path.resolve()

    def clean(self) -> Path:
        """Remove the SDK directory.

        Returns:
            The path that was removed.

        Raises:
            RuntimeError: If SDK path is not configured.
        """
        if self._sdk_path is None:
            _log.error("Cannot clean: SDK path not configured")
            raise RuntimeError(_ERR_NOT_INSTALLED_M2A)
        path = self._sdk_path
        _log.info(f"Removing QAIRT SDK at {path}")
        shutil.rmtree(path, ignore_errors=True)
        _log.debug(f"SDK directory removed: {path}")
        self._sdk_path = None
        return path

    # --- Internal ---

    def _cleanup_zip_files(self) -> None:
        """Remove any .zip files from the install directory after extraction."""
        for zip_file in self._install_dir.glob("*.zip"):
            try:
                _log.debug(f"Removing zip file: {zip_file}")
                zip_file.unlink()
            except OSError as e:  # noqa: PERF203
                _log.warning(f"Failed to remove zip file {zip_file}: {e}")

    def _find_sdk_path(self) -> Path | None:
        """Glob install_dir/qairt/<version>.* and return the first match."""
        base = self._install_dir / "qairt"
        if not base.exists():
            _log.debug(f"QAIRT base directory does not exist: {base}")
            return None
        matches = sorted(base.glob(f"{self._sdk_version}.*"))
        if matches:
            path = matches[0]
            _log.debug(f"Found SDK path: {path}")
            return path
        _log.warning(f"No SDK path found matching version {self._sdk_version} in {base}")
        return None
