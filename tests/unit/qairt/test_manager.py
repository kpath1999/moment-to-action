"""Unit tests for QairtSDKManager."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from moment_to_action.config import AppConfig
from moment_to_action.qairt._manager import QairtSDKManager


def _make_mgr(
    sdk_path: Path | None = None,
    sdk_version: str = "2.45.0",
    install_dir: Path | None = None,
    tmp_path: Path | None = None,
) -> QairtSDKManager:
    if install_dir is None:
        install_dir = tmp_path or Path("/tmp/test_install")
    return QairtSDKManager(sdk_path=sdk_path, sdk_version=sdk_version, install_dir=install_dir)


@pytest.mark.unit
class TestQairtSDKManagerFromAppConfig:
    """Tests for QairtSDKManager.from_app_config."""

    def test_from_app_config(self, tmp_path: Path) -> None:
        """from_app_config populates sdk_path and sdk_version from config."""
        config = AppConfig(qairt_sdk_path=tmp_path / "sdk", qairt_sdk_version="2.44.0")
        mock_pm = MagicMock()
        mock_pm.data.data_dir = tmp_path
        mgr = QairtSDKManager.from_app_config(config, mock_pm)
        assert mgr.path == tmp_path / "sdk"
        assert mgr.configured_version == "2.44.0"

    def test_from_app_config_no_path(self, tmp_path: Path) -> None:
        """from_app_config with no qairt_sdk_path yields path=None."""
        config = AppConfig()
        mock_pm = MagicMock()
        mock_pm.data.data_dir = tmp_path
        mgr = QairtSDKManager.from_app_config(config, mock_pm)
        assert mgr.path is None


@pytest.mark.unit
class TestQairtSDKManagerProperties:
    """Tests for QairtSDKManager read-only properties."""

    def test_is_available_false_when_path_none(self) -> None:
        """is_available is False when no path configured."""
        mgr = _make_mgr(sdk_path=None)
        assert mgr.is_available is False

    def test_is_available_false_when_dir_missing(self, tmp_path: Path) -> None:
        """is_available is False when configured path does not exist on disk."""
        mgr = _make_mgr(sdk_path=tmp_path / "nonexistent")
        assert mgr.is_available is False

    def test_is_available_true_when_dir_exists(self, tmp_path: Path) -> None:
        """is_available is True when configured path exists on disk."""
        sdk = tmp_path / "2.45.0.24"
        sdk.mkdir()
        mgr = _make_mgr(sdk_path=sdk)
        assert mgr.is_available is True

    def test_path_returns_configured(self, tmp_path: Path) -> None:
        """Path property returns the configured SDK path."""
        p = tmp_path / "sdk"
        mgr = _make_mgr(sdk_path=p)
        assert mgr.path == p

    def test_configured_version(self) -> None:
        """configured_version returns the version string from construction."""
        mgr = _make_mgr(sdk_version="2.44.0")
        assert mgr.configured_version == "2.44.0"

    def test_installed_version_none_when_unavailable(self) -> None:
        """installed_version is None when SDK not configured."""
        mgr = _make_mgr(sdk_path=None)
        assert mgr.installed_version is None

    def test_installed_version_none_when_dir_missing(self, tmp_path: Path) -> None:
        """installed_version is None when configured path does not exist."""
        mgr = _make_mgr(sdk_path=tmp_path / "nonexistent")
        assert mgr.installed_version is None

    def test_installed_version_from_path_name(self, tmp_path: Path) -> None:
        """installed_version is derived from the directory name."""
        sdk = tmp_path / "2.45.0.24"
        sdk.mkdir()
        mgr = _make_mgr(sdk_path=sdk)
        assert mgr.installed_version == "2.45.0.24"


@pytest.mark.unit
class TestQairtSDKManagerConfigureEnv:
    """Tests for QairtSDKManager.configure_env."""

    def test_configure_env_raises_when_no_path(self) -> None:
        """configure_env raises RuntimeError when SDK not installed."""
        mgr = _make_mgr(sdk_path=None)
        with pytest.raises(RuntimeError, match="not installed"):
            mgr.configure_env()

    def test_configure_env_sets_env_var(self, tmp_path: Path) -> None:
        """configure_env sets QAIRT_SDK_ROOT to the installed path."""
        sdk = tmp_path / "2.45.0.24"
        sdk.mkdir()
        mgr = _make_mgr(sdk_path=sdk)
        old = os.environ.pop("QAIRT_SDK_ROOT", None)
        try:
            mgr.configure_env()
            assert os.environ["QAIRT_SDK_ROOT"] == str(sdk)
        finally:
            if old is None:
                os.environ.pop("QAIRT_SDK_ROOT", None)
            else:
                os.environ["QAIRT_SDK_ROOT"] = old


@pytest.mark.unit
class TestQairtSDKManagerCheckDeps:
    """Tests for QairtSDKManager.check_system_deps."""

    def test_delegates_to_deps_module(self) -> None:
        """check_system_deps delegates to the _deps.check_system_deps function."""
        mgr = _make_mgr()
        with patch(
            "moment_to_action.qairt._manager.QairtSDKManager.check_system_deps",
            return_value=["clang"],
        ):
            result = mgr.check_system_deps()
        assert result == ["clang"]

    def test_returns_empty_when_all_deps_present(self) -> None:
        """Returns empty list when check_system_deps reports no missing packages."""
        mgr = _make_mgr()
        with patch("moment_to_action.qairt._deps.check_system_deps", return_value=[]):
            result = mgr.check_system_deps()
        assert result == []


@pytest.mark.unit
class TestQairtSDKManagerInstall:
    """Tests for QairtSDKManager.install."""

    def test_install_success(self, tmp_path: Path) -> None:
        """Install returns the extracted SDK path on success."""
        sdk = tmp_path / "qairt" / "2.45.0.24"
        sdk.mkdir(parents=True)
        mgr = _make_mgr(install_dir=tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("moment_to_action.qairt._manager.subprocess.run", return_value=mock_result):
            path = mgr.install(stream=False)

        assert path == sdk
        assert mgr.path == sdk

    def test_install_calls_fetch_with_correct_args(self, tmp_path: Path) -> None:
        """Install invokes qairt-vm fetch with version and dir flags."""
        sdk = tmp_path / "qairt" / "2.45.0.24"
        sdk.mkdir(parents=True)
        mgr = _make_mgr(sdk_version="2.45.0", install_dir=tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch(
            "moment_to_action.qairt._manager.subprocess.run", return_value=mock_result
        ) as mock_run:
            mgr.install(stream=False)

        args = mock_run.call_args[0][0]
        assert "fetch" in args
        assert "--version" in args
        assert "2.45.0" in args
        assert "--dir" in args
        assert str(tmp_path) in args

    def test_install_raises_on_nonzero_returncode(self, tmp_path: Path) -> None:
        """Install raises RuntimeError when qairt-vm exits non-zero."""
        mgr = _make_mgr(install_dir=tmp_path)
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("moment_to_action.qairt._manager.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="fetch failed"):
                mgr.install(stream=False)

    def test_install_raises_when_path_not_found(self, tmp_path: Path) -> None:
        """Install raises RuntimeError when extracted path cannot be found."""
        mgr = _make_mgr(install_dir=tmp_path)
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("moment_to_action.qairt._manager.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="SDK path not found"):
                mgr.install(stream=False)

    def test_install_stream_true_sets_capture_false(self, tmp_path: Path) -> None:
        """Install with stream=True passes capture_output=False to subprocess."""
        sdk = tmp_path / "qairt" / "2.45.0.24"
        sdk.mkdir(parents=True)
        mgr = _make_mgr(install_dir=tmp_path)

        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch(
            "moment_to_action.qairt._manager.subprocess.run", return_value=mock_result
        ) as mock_run:
            mgr.install(stream=True)

        assert mock_run.call_args.kwargs.get("capture_output") is False

    def test_install_raises_when_already_installed(self, tmp_path: Path) -> None:
        """Install raises RuntimeError when SDK is already available."""
        sdk = tmp_path / "2.45.0.24"
        sdk.mkdir(parents=True)
        mgr = _make_mgr(sdk_path=sdk, install_dir=tmp_path)
        with pytest.raises(RuntimeError, match="already installed"):
            mgr.install(stream=False)


@pytest.mark.unit
class TestQairtSDKManagerCleanupZips:
    """Tests for QairtSDKManager._cleanup_zip_files."""

    def test_cleanup_removes_zip_files(self, tmp_path: Path) -> None:
        """_cleanup_zip_files removes .zip files from install directory."""
        (tmp_path / "2.45.0.260326.zip").touch()
        (tmp_path / "other.zip").touch()
        mgr = _make_mgr(install_dir=tmp_path)
        mgr._cleanup_zip_files()
        assert not (tmp_path / "2.45.0.260326.zip").exists()
        assert not (tmp_path / "other.zip").exists()

    def test_cleanup_ignores_non_zip_files(self, tmp_path: Path) -> None:
        """_cleanup_zip_files only removes .zip files."""
        (tmp_path / "file.txt").touch()
        (tmp_path / "archive.tar.gz").touch()
        mgr = _make_mgr(install_dir=tmp_path)
        mgr._cleanup_zip_files()
        assert (tmp_path / "file.txt").exists()
        assert (tmp_path / "archive.tar.gz").exists()

    def test_cleanup_handles_missing_files(self, tmp_path: Path) -> None:
        """_cleanup_zip_files handles missing files gracefully."""
        mgr = _make_mgr(install_dir=tmp_path)
        mgr._cleanup_zip_files()
        assert True


@pytest.mark.unit
class TestQairtSDKManagerFindSdkPath:
    """Tests for QairtSDKManager._find_sdk_path."""

    def test_find_sdk_path_returns_none_when_qairt_dir_missing(self, tmp_path: Path) -> None:
        """_find_sdk_path returns None when qairt directory does not exist."""
        mgr = _make_mgr(install_dir=tmp_path)
        path = mgr._find_sdk_path()
        assert path is None

    def test_find_sdk_path_returns_none_when_no_version_match(self, tmp_path: Path) -> None:
        """_find_sdk_path returns None when no versions match the pattern."""
        base = tmp_path / "qairt"
        base.mkdir()
        (base / "2.44.0.24").mkdir()
        mgr = _make_mgr(sdk_version="2.45.0", install_dir=tmp_path)
        path = mgr._find_sdk_path()
        assert path is None

    def test_find_sdk_path_returns_first_match(self, tmp_path: Path) -> None:
        """_find_sdk_path returns the first matching version directory."""
        base = tmp_path / "qairt"
        base.mkdir()
        (base / "2.45.0.20").mkdir()
        (base / "2.45.0.24").mkdir()
        mgr = _make_mgr(sdk_version="2.45.0", install_dir=tmp_path)
        path = mgr._find_sdk_path()
        assert path == base / "2.45.0.20"


@pytest.mark.unit
class TestQairtSDKManagerVerify:
    """Tests for QairtSDKManager.verify."""

    def test_verify_raises_when_no_path(self) -> None:
        """Verify raises RuntimeError when SDK not installed."""
        mgr = _make_mgr(sdk_path=None)
        with pytest.raises(RuntimeError, match="not installed"):
            mgr.verify()

    def test_verify_returns_empty_on_ubuntu_when_ok(self, tmp_path: Path) -> None:
        """On Ubuntu (dpkg present), returns empty list when all checks pass."""
        sdk = tmp_path / "2.45.0.24"
        sdk.mkdir()
        mgr = _make_mgr(sdk_path=sdk)
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("moment_to_action.qairt._deps.check_system_deps", return_value=[]):
            with patch(
                "moment_to_action.qairt._manager.shutil.which", return_value="/usr/bin/dpkg"
            ):
                with patch(
                    "moment_to_action.qairt._manager.subprocess.run", return_value=mock_result
                ):
                    result = mgr.verify(stream=False)
        assert result == []

    def test_verify_runs_inspect_on_ubuntu(self, tmp_path: Path) -> None:
        """On Ubuntu, verify calls qairt-vm --inspect with QAIRT_SDK_ROOT set."""
        sdk = tmp_path / "2.45.0.24"
        sdk.mkdir()
        mgr = _make_mgr(sdk_path=sdk)
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("moment_to_action.qairt._deps.check_system_deps", return_value=[]):
            with patch(
                "moment_to_action.qairt._manager.shutil.which", return_value="/usr/bin/dpkg"
            ):
                with patch(
                    "moment_to_action.qairt._manager.subprocess.run", return_value=mock_result
                ) as mock_run:
                    mgr.verify(stream=False)
        args = mock_run.call_args[0][0]
        assert "--inspect" in args
        env_passed = mock_run.call_args.kwargs["env"]
        assert env_passed["QAIRT_SDK_ROOT"] == str(sdk)

    def test_verify_inspect_failure_adds_warning(self, tmp_path: Path) -> None:
        """Non-zero --inspect exit code is included as a warning."""
        sdk = tmp_path / "2.45.0.24"
        sdk.mkdir()
        mgr = _make_mgr(sdk_path=sdk)
        mock_result = MagicMock()
        mock_result.returncode = 2
        with patch("moment_to_action.qairt._deps.check_system_deps", return_value=[]):
            with patch(
                "moment_to_action.qairt._manager.shutil.which", return_value="/usr/bin/dpkg"
            ):
                with patch(
                    "moment_to_action.qairt._manager.subprocess.run", return_value=mock_result
                ):
                    result = mgr.verify(stream=False)
        assert any("--inspect failed" in w for w in result)

    def test_verify_warns_unsupported_system(self, tmp_path: Path) -> None:
        """On non-Ubuntu, verify adds an unsupported-system warning, skips inspect."""
        sdk = tmp_path / "2.45.0.24"
        sdk.mkdir()
        mgr = _make_mgr(sdk_path=sdk)
        with patch("moment_to_action.qairt._deps.check_system_deps", return_value=[]):
            with patch("moment_to_action.qairt._manager.shutil.which", return_value=None):
                result = mgr.verify(stream=False)
        assert any("officially supported on Ubuntu" in w for w in result)

    def test_verify_warns_when_path_missing(self, tmp_path: Path) -> None:
        """Verify includes warning when configured SDK path does not exist on disk."""
        sdk = tmp_path / "2.45.0.24"
        mgr = _make_mgr(sdk_path=sdk)
        with patch("moment_to_action.qairt._deps.check_system_deps", return_value=[]):
            with patch("moment_to_action.qairt._manager.shutil.which", return_value=None):
                result = mgr.verify(stream=False)
        assert any("does not exist" in w for w in result)

    def test_verify_warns_missing_deps(self, tmp_path: Path) -> None:
        """Verify includes one warning per missing system package."""
        sdk = tmp_path / "2.45.0.24"
        sdk.mkdir()
        mgr = _make_mgr(sdk_path=sdk)
        with patch(
            "moment_to_action.qairt._deps.check_system_deps", return_value=["clang", "curl"]
        ):
            with patch("moment_to_action.qairt._manager.shutil.which", return_value=None):
                result = mgr.verify(stream=False)
        pkg_warnings = [w for w in result if "Missing system package" in w]
        assert len(pkg_warnings) == 2


@pytest.mark.unit
class TestQairtSDKManagerClean:
    """Tests for QairtSDKManager.clean."""

    def test_clean_raises_when_no_path(self) -> None:
        """Clean raises RuntimeError when SDK not installed via m2a."""
        mgr = _make_mgr(sdk_path=None)
        with pytest.raises(RuntimeError, match="not installed"):
            mgr.clean()

    def test_clean_calls_rmtree(self, tmp_path: Path) -> None:
        """Clean calls shutil.rmtree on the SDK directory."""
        sdk = tmp_path / "2.45.0.24"
        sdk.mkdir()
        mgr = _make_mgr(sdk_path=sdk)
        with patch("moment_to_action.qairt._manager.shutil.rmtree") as mock_rmtree:
            mgr.clean()
        mock_rmtree.assert_called_once_with(sdk, ignore_errors=True)

    def test_clean_returns_removed_path(self, tmp_path: Path) -> None:
        """Clean returns the path that was removed."""
        sdk = tmp_path / "2.45.0.24"
        sdk.mkdir()
        mgr = _make_mgr(sdk_path=sdk)
        with patch("moment_to_action.qairt._manager.shutil.rmtree"):
            removed = mgr.clean()
        assert removed == sdk

    def test_clean_clears_internal_path(self, tmp_path: Path) -> None:
        """Clean sets the internal sdk_path to None."""
        sdk = tmp_path / "2.45.0.24"
        sdk.mkdir()
        mgr = _make_mgr(sdk_path=sdk)
        with patch("moment_to_action.qairt._manager.shutil.rmtree"):
            mgr.clean()
        assert mgr.path is None
