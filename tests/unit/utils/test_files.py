"""Unit tests for moment_to_action.utils.files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from moment_to_action.utils.files import clear_directory, disk_size

if TYPE_CHECKING:
    from pathlib import Path


def _write_file(path: Path, payload: bytes) -> int:
    """Create `path` with the given bytes; return the byte count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return len(payload)


@pytest.mark.unit
class TestDiskSize:
    """Tests for `disk_size`."""

    def test_file_returns_size(self, tmp_path: Path) -> None:
        """disk_size on a file returns its size in bytes."""
        f = tmp_path / "f.bin"
        size = _write_file(f, b"x" * 42)
        assert disk_size(f) == size

    def test_empty_dir_returns_zero(self, tmp_path: Path) -> None:
        """disk_size on an empty directory returns 0."""
        d = tmp_path / "empty"
        d.mkdir()
        assert disk_size(d) == 0

    def test_dir_sums_recursive_file_sizes(self, tmp_path: Path) -> None:
        """disk_size on a directory recurses and sums all file sizes."""
        a = _write_file(tmp_path / "a.bin", b"a" * 10)
        b = _write_file(tmp_path / "nested" / "b.bin", b"b" * 25)
        c = _write_file(tmp_path / "nested" / "deep" / "c.bin", b"c" * 7)
        assert disk_size(tmp_path) == a + b + c

    def test_missing_path_raises_value_error(self, tmp_path: Path) -> None:
        """disk_size on a non-existent path raises ValueError."""
        ghost = tmp_path / "ghost"
        with pytest.raises(ValueError, match=r"neither a file nor a directory"):
            disk_size(ghost)


@pytest.mark.unit
class TestClearDirectory:
    """Tests for `clear_directory`."""

    def test_returns_zero_for_empty_dir(self, tmp_path: Path) -> None:
        """Clearing an empty directory returns 0 and removes the dir."""
        d = tmp_path / "empty"
        d.mkdir()
        assert clear_directory(d) == 0
        assert not d.exists()

    def test_clears_flat_dir(self, tmp_path: Path) -> None:
        """Clearing a directory with only files returns total bytes and removes the dir."""
        d = tmp_path / "flat"
        d.mkdir()
        size = _write_file(d / "a", b"a" * 100) + _write_file(d / "b", b"b" * 200)
        assert clear_directory(d) == size
        assert not d.exists()

    def test_clears_nested_dirs(self, tmp_path: Path) -> None:
        """Clearing a directory recursively unlinks all files and removes dirs."""
        d = tmp_path / "nested"
        size = 0
        size += _write_file(d / "a.bin", b"a" * 1)
        size += _write_file(d / "sub" / "b.bin", b"b" * 2)
        size += _write_file(d / "sub" / "deeper" / "c.bin", b"c" * 3)

        assert clear_directory(d) == size
        assert not d.exists()
