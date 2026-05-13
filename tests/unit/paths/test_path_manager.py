"""Unit tests for moment_to_action.paths._manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from moment_to_action._version import VERSION
from moment_to_action.paths import CacheManager, PathManager
from moment_to_action.paths._data._manager import DataManager

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class _FakeDirs:
    """Minimal stand-in for platformdirs.PlatformDirs."""

    user_cache_path: Path
    user_data_path: Path
    user_log_path: Path
    user_config_path: Path


def _patch_platform_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    captured: list[dict[str, Any]] | None = None,
) -> _FakeDirs:
    """Patch `PlatformDirs` in the path manager module.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        tmp_path: temp dir to anchor fake paths in.
        captured: if provided, kwargs passed to `PlatformDirs` are appended to it.

    Returns:
        The `_FakeDirs` instance the patched factory will return.
    """
    fake = _FakeDirs(
        user_cache_path=tmp_path / "cache",
        user_data_path=tmp_path / "data",
        user_log_path=tmp_path / "logs",
        user_config_path=tmp_path / "config",
    )

    def factory(**kwargs: Any) -> _FakeDirs:
        if captured is not None:
            captured.append(kwargs)
        return fake

    monkeypatch.setattr("moment_to_action.paths._manager.PlatformDirs", factory)
    return fake


@pytest.mark.unit
class TestInit:
    """Tests for PathManager.__init__."""

    def test_creates_cache_and_data_managers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Construction wires up CacheManager and DataManager on the fake platform paths."""
        fake = _patch_platform_dirs(monkeypatch, tmp_path)
        pm = PathManager()
        assert isinstance(pm.cache, CacheManager)
        assert pm.cache.cache_dir == fake.user_cache_path
        assert isinstance(pm.data, DataManager)
        assert pm.data.data_dir == fake.user_data_path

    def test_forwards_defaults_to_platformdirs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default app name, author, version, and `ensure_exists` are passed through."""
        captured: list[dict[str, Any]] = []
        _patch_platform_dirs(monkeypatch, tmp_path, captured=captured)
        PathManager()
        assert captured == [
            {
                "appname": "MomentToAction",
                "appauthor": "GeorgiaTech",
                "version": VERSION,
                "ensure_exists": True,
            }
        ]

    def test_forwards_custom_app_and_author(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Custom app name and author are forwarded to PlatformDirs."""
        captured: list[dict[str, Any]] = []
        _patch_platform_dirs(monkeypatch, tmp_path, captured=captured)
        PathManager(app_name="OtherApp", author="OtherOrg")
        assert captured[0]["appname"] == "OtherApp"
        assert captured[0]["appauthor"] == "OtherOrg"


@pytest.mark.unit
class TestProperties:
    """Tests for PathManager properties."""

    def test_cache_returns_cache_manager(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`cache` returns the CacheManager built at construction."""
        _patch_platform_dirs(monkeypatch, tmp_path)
        pm = PathManager()
        assert pm.cache is pm.cache
        assert isinstance(pm.cache, CacheManager)

    def test_data_returns_data_manager(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`data` returns the DataManager built at construction."""
        _patch_platform_dirs(monkeypatch, tmp_path)
        pm = PathManager()
        assert pm.data is pm.data
        assert isinstance(pm.data, DataManager)

    def test_logs_dir_returns_user_log_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`logs_dir` returns the platform-provided user log path."""
        fake = _patch_platform_dirs(monkeypatch, tmp_path)
        pm = PathManager()
        assert pm.logs_dir == fake.user_log_path

    def test_app_config_file_appends_config_json(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`app_config_file` returns `<user_config_path>/config.json`."""
        fake = _patch_platform_dirs(monkeypatch, tmp_path)
        pm = PathManager()
        assert pm.app_config_file == fake.user_config_path / "config.json"
