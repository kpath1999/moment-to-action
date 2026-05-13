"""Unit tests for moment_to_action.paths._cache._manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from moment_to_action.paths._cache._manager import CacheInfo, CacheManager
from moment_to_action.paths._cache._models import (
    CachedModelInfo,
    ModelCacheContents,
    ModelCacheManager,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_file(path: Path, payload: bytes) -> int:
    """Create `path` with the given bytes; return the byte count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return len(payload)


# ---------------------------------------------------------------------------
# CacheInfo
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCacheInfo:
    """Tests for CacheInfo."""

    def test_to_json_serializes_paths_and_models(self, tmp_path: Path) -> None:
        """to_json yields total size, string root paths, and models_info dict."""
        stray = tmp_path / "stray.bin"
        models_info = ModelCacheContents(
            total_size_bytes=42,
            models={"yolo": CachedModelInfo("yolo", 42, ["fp32"], [])},
            other=[],
        )
        info = CacheInfo(
            total_size_bytes=100,
            root_contents=[stray],
            models_info=models_info,
        )

        as_json = info.to_json()
        models_info_json = as_json["models_info"]
        assert as_json["total_size_bytes"] == 100
        assert as_json["root_contents"] == [str(stray)]
        assert isinstance(models_info_json, dict)
        assert models_info_json["total_size_bytes"] == 42

    def test_to_rich_table_includes_models_subcache(self) -> None:
        """The rich table includes the 'models' subcache row."""
        info = CacheInfo(
            total_size_bytes=10,
            root_contents=[],
            models_info=ModelCacheContents(total_size_bytes=10, models={}, other=[]),
        )
        table = info.to_rich_table()
        # One row for the models subcache; no `other` row since root_contents is empty.
        assert table.row_count == 1

    def test_to_rich_table_appends_other_row(self, tmp_path: Path) -> None:
        """When root_contents is non-empty, an 'other' row is appended."""
        info = CacheInfo(
            total_size_bytes=10,
            root_contents=[tmp_path / "stray.bin"],
            models_info=ModelCacheContents(total_size_bytes=0, models={}, other=[]),
        )
        table = info.to_rich_table()
        # models row + other row
        assert table.row_count == 2


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInit:
    """Tests for CacheManager.__init__."""

    def test_creates_cache_dir_when_missing(self, tmp_path: Path) -> None:
        """The cache directory is created on construction."""
        target = tmp_path / "cache"
        assert not target.exists()
        CacheManager(target)
        assert target.is_dir()

    def test_initialises_model_submanager(self, tmp_path: Path) -> None:
        """A ModelCacheManager is created and anchored at `<cache>/models`."""
        cache = CacheManager(tmp_path / "cache")
        assert isinstance(cache.models, ModelCacheManager)
        assert cache.models.models_dir == tmp_path / "cache" / "models"
        assert (tmp_path / "cache" / "models").is_dir()


@pytest.mark.unit
class TestProperties:
    """Tests for CacheManager.cache_dir and CacheManager.models."""

    def test_cache_dir_returns_configured_path(self, tmp_path: Path) -> None:
        """cache_dir returns the path supplied at construction."""
        cache = CacheManager(tmp_path / "cache")
        assert cache.cache_dir == tmp_path / "cache"

    def test_models_returns_same_instance(self, tmp_path: Path) -> None:
        """The same ModelCacheManager instance is returned on each access."""
        cache = CacheManager(tmp_path / "cache")
        assert cache.models is cache.models


@pytest.mark.unit
class TestInspectCache:
    """Tests for CacheManager.inspect_cache."""

    def test_empty_cache(self, tmp_path: Path) -> None:
        """Inspecting an empty cache reports zero size, no models, no root contents."""
        cache = CacheManager(tmp_path / "cache")
        info = cache.inspect_cache()
        assert info.total_size_bytes == 0
        assert info.root_contents == []
        assert info.models_info.total_size_bytes == 0

    def test_sums_root_and_model_size_without_recomputing_models(
        self,
        tmp_path: Path,
    ) -> None:
        """`total_size_bytes` = sum(disk_size(root files)) + models.total_size_bytes (cached).

        Verifies the additive shortcut from `_manager.py` does NOT double-count or recurse
        into the models subtree when it already has a computed size.
        """
        cache = CacheManager(tmp_path / "cache")
        stray_size = _write_file(tmp_path / "cache" / "stray.txt", b"x" * 31)
        model_size = _write_file(
            tmp_path / "cache" / "models" / "yolo" / "fp32" / "w.bin",
            b"y" * 77,
        )

        info = cache.inspect_cache()

        assert info.total_size_bytes == stray_size + model_size
        assert info.models_info.total_size_bytes == model_size
        assert (tmp_path / "cache" / "stray.txt") in info.root_contents

    def test_excludes_models_dir_from_root_contents(self, tmp_path: Path) -> None:
        """The 'models' sub-directory is never listed as a root content."""
        cache = CacheManager(tmp_path / "cache")
        _write_file(
            tmp_path / "cache" / "models" / "yolo" / "fp32" / "w.bin",
            b"y" * 5,
        )
        info = cache.inspect_cache()
        assert info.root_contents == []

    def test_size_is_pure_sum_no_double_counting(self, tmp_path: Path) -> None:
        """With multiple models + multiple stray files, total equals exact sum."""
        cache = CacheManager(tmp_path / "cache")
        a = _write_file(tmp_path / "cache" / "models" / "yolo" / "fp32" / "w.bin", b"a" * 11)
        b = _write_file(tmp_path / "cache" / "models" / "yolo" / "int8" / "w.bin", b"b" * 17)
        c = _write_file(tmp_path / "cache" / "models" / "clip" / "fp16" / "w.bin", b"c" * 13)
        d = _write_file(tmp_path / "cache" / "f1.bin", b"d" * 5)
        e = _write_file(tmp_path / "cache" / "sub" / "f2.bin", b"e" * 19)

        info = cache.inspect_cache()
        assert info.total_size_bytes == a + b + c + d + e


@pytest.mark.unit
class TestClearCache:
    """Tests for CacheManager.clear_cache."""

    def test_clears_models_only(self, tmp_path: Path) -> None:
        """When only the models submanager has content, only its size is reported."""
        cache = CacheManager(tmp_path / "cache")
        size = _write_file(tmp_path / "cache" / "models" / "yolo" / "fp32" / "w.bin", b"x" * 9)

        info = cache.clear_cache()

        assert info.total_size_bytes == size
        assert info.models_info.total_size_bytes == size
        # The model submanager removes its own root; cache_dir itself remains.
        assert not (tmp_path / "cache" / "models").exists()
        assert (tmp_path / "cache").is_dir()

    def test_clears_unexpected_file_with_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Unexpected file in cache root is unlinked, counted, and warning is logged."""
        import logging

        cache = CacheManager(tmp_path / "cache")
        stray = tmp_path / "cache" / "stray.txt"
        stray_size = _write_file(stray, b"y" * 17)

        with caplog.at_level(logging.WARNING, logger="moment_to_action.paths._cache._manager"):
            info = cache.clear_cache()

        assert info.total_size_bytes == stray_size
        assert stray in info.root_contents
        assert not stray.exists()
        assert any(
            "stray.txt" in record.getMessage() and record.levelno == logging.WARNING
            for record in caplog.records
        )

    def test_clears_unexpected_dir_with_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An unexpected subdir is cleared recursively, counted, and warning logged."""
        import logging

        cache = CacheManager(tmp_path / "cache")
        stray_dir = tmp_path / "cache" / "weird"
        stray_size = _write_file(stray_dir / "nested" / "blob.bin", b"z" * 21)

        with caplog.at_level(logging.WARNING, logger="moment_to_action.paths._cache._manager"):
            info = cache.clear_cache()

        assert info.total_size_bytes == stray_size
        assert stray_dir in info.root_contents
        assert not stray_dir.exists()
        assert any(
            "weird" in record.getMessage() and record.levelno == logging.WARNING
            for record in caplog.records
        )

    def test_combined_models_and_stray_size_sum(self, tmp_path: Path) -> None:
        """Total cleared equals model contents + stray entries at root."""
        cache = CacheManager(tmp_path / "cache")
        model_size = _write_file(
            tmp_path / "cache" / "models" / "yolo" / "fp32" / "w.bin", b"a" * 19
        )
        stray_size = _write_file(tmp_path / "cache" / "stray.bin", b"b" * 23)

        info = cache.clear_cache()

        assert info.total_size_bytes == model_size + stray_size

    def test_empty_cache_clears_to_zero(self, tmp_path: Path) -> None:
        """Clearing an empty cache returns a zero-sized CacheInfo."""
        cache = CacheManager(tmp_path / "cache")
        info = cache.clear_cache()
        assert info.total_size_bytes == 0
        assert info.root_contents == []
        assert info.models_info.total_size_bytes == 0
