"""Unit tests for moment_to_action.paths._cache._manager."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from moment_to_action.paths._cache._manager import CacheManager
from moment_to_action.paths._cache._models import ModelCacheManager

if TYPE_CHECKING:
    from pathlib import Path


def _write_file(path: Path, payload: bytes) -> int:
    """Create `path` with the given bytes; return the byte count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return len(payload)


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
class TestClearCache:
    """Tests for CacheManager.clear_cache."""

    def test_clears_models_only(self, tmp_path: Path) -> None:
        """When only the models submanager has content, only its size is reported."""
        cache = CacheManager(tmp_path / "cache")
        size = _write_file(tmp_path / "cache" / "models" / "yolo" / "fp32" / "w.bin", b"x" * 9)

        freed = cache.clear_cache()

        assert freed == size
        # The model submanager removes its own root; cache_dir itself remains.
        assert not (tmp_path / "cache" / "models").exists()
        assert (tmp_path / "cache").is_dir()

    def test_clears_unexpected_file(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Unexpected file in the cache root is unlinked, counted, and warning is logged."""
        cache = CacheManager(tmp_path / "cache")
        stray = tmp_path / "cache" / "stray.txt"
        stray_size = _write_file(stray, b"y" * 17)

        with caplog.at_level(logging.WARNING, logger="moment_to_action.paths._cache._manager"):
            freed = cache.clear_cache()

        assert freed == stray_size
        assert not stray.exists()
        assert any(
            "stray.txt" in record.getMessage() and record.levelno == logging.WARNING
            for record in caplog.records
        )

    def test_clears_unexpected_dir(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An unexpected subdir is cleared recursively, its size counted, and a warning logged."""
        cache = CacheManager(tmp_path / "cache")
        stray_dir = tmp_path / "cache" / "weird"
        stray_size = _write_file(stray_dir / "nested" / "blob.bin", b"z" * 21)

        with caplog.at_level(logging.WARNING, logger="moment_to_action.paths._cache._manager"):
            freed = cache.clear_cache()

        assert freed == stray_size
        assert not stray_dir.exists()
        assert any(
            "weird" in record.getMessage() and record.levelno == logging.WARNING
            for record in caplog.records
        )
