"""Unit tests for moment_to_action.paths._data._manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from moment_to_action.paths._data._manager import DataManager

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
class TestInit:
    """Tests for DataManager.__init__."""

    def test_creates_data_dir_when_missing(self, tmp_path: Path) -> None:
        """A non-existent data dir is created on construction."""
        target = tmp_path / "fresh"
        assert not target.exists()
        DataManager(target)
        assert target.is_dir()

    def test_accepts_existing_data_dir(self, tmp_path: Path) -> None:
        """Construction does not raise when the data dir already exists."""
        target = tmp_path / "existing"
        target.mkdir()
        DataManager(target)  # must not raise
        assert target.is_dir()


@pytest.mark.unit
class TestDataDir:
    """Tests for DataManager.data_dir."""

    def test_returns_configured_path(self, tmp_path: Path) -> None:
        """data_dir returns the path supplied at construction time."""
        mgr = DataManager(tmp_path / "d")
        assert mgr.data_dir == tmp_path / "d"


@pytest.mark.unit
class TestQairtDir:
    """Tests for DataManager.qairt_dir."""

    def test_lazy_creates_subdir(self, tmp_path: Path) -> None:
        """First access creates `<data>/qairt`."""
        mgr = DataManager(tmp_path / "d")
        assert not (tmp_path / "d" / "qairt").exists()
        path = mgr.qairt_dir
        assert path == tmp_path / "d" / "qairt"
        assert path.is_dir()

    def test_repeated_access_is_idempotent(self, tmp_path: Path) -> None:
        """Repeated access does not raise even though the directory already exists."""
        mgr = DataManager(tmp_path / "d")
        first = mgr.qairt_dir
        second = mgr.qairt_dir
        assert first == second
        assert first.is_dir()
