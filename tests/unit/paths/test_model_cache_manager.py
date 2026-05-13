"""Unit tests for moment_to_action.paths._cache._models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

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
    """Tests for ModelCacheManager.__init__."""

    def test_creates_dir_when_missing(self, tmp_path: Path) -> None:
        """Construction creates the model cache dir."""
        target = tmp_path / "models"
        assert not target.exists()
        ModelCacheManager(target)
        assert target.is_dir()

    def test_accepts_existing_dir(self, tmp_path: Path) -> None:
        """Construction is idempotent when the dir already exists."""
        target = tmp_path / "models"
        target.mkdir()
        ModelCacheManager(target)  # must not raise
        assert target.is_dir()


@pytest.mark.unit
class TestModelsDir:
    """Tests for ModelCacheManager.models_dir."""

    def test_returns_configured_path(self, tmp_path: Path) -> None:
        """models_dir returns the path passed at construction."""
        mgr = ModelCacheManager(tmp_path / "models")
        assert mgr.models_dir == tmp_path / "models"


@pytest.mark.unit
class TestGetModelDir:
    """Tests for ModelCacheManager.get_model_dir."""

    def test_create_true_makes_variant_dir(self, tmp_path: Path) -> None:
        """create=True creates the variant directory."""
        mgr = ModelCacheManager(tmp_path / "models")
        path = mgr.get_model_dir("yolo", "fp32", create=True)
        assert path == tmp_path / "models" / "yolo" / "fp32"
        assert path.is_dir()

    def test_create_true_is_idempotent(self, tmp_path: Path) -> None:
        """create=True is a no-op when the directory already exists."""
        mgr = ModelCacheManager(tmp_path / "models")
        first = mgr.get_model_dir("yolo", "fp32", create=True)
        second = mgr.get_model_dir("yolo", "fp32", create=True)
        assert first == second
        assert first.is_dir()

    def test_missing_raises_with_model_message(self, tmp_path: Path) -> None:
        """When neither the model nor the variant exists, the error mentions the model dir."""
        mgr = ModelCacheManager(tmp_path / "models")
        with pytest.raises(FileNotFoundError, match=r"Model directory .* does not exist"):
            mgr.get_model_dir("yolo", "fp32")

    def test_missing_raises_with_variant_message(self, tmp_path: Path) -> None:
        """When the model exists but the variant does not, the error mentions the variant dir."""
        mgr = ModelCacheManager(tmp_path / "models")
        (tmp_path / "models" / "yolo").mkdir()
        with pytest.raises(FileNotFoundError, match=r"Model variant directory .* does not exist"):
            mgr.get_model_dir("yolo", "fp32")

    def test_returns_existing_dir_without_creating(self, tmp_path: Path) -> None:
        """create=False returns the path when the variant already exists."""
        mgr = ModelCacheManager(tmp_path / "models")
        expected = tmp_path / "models" / "yolo" / "fp32"
        expected.mkdir(parents=True)
        result = mgr.get_model_dir("yolo", "fp32")
        assert result == expected


@pytest.mark.unit
class TestIsCached:
    """Tests for ModelCacheManager.is_cached."""

    def test_with_variant_true(self, tmp_path: Path) -> None:
        """Returns True when the requested variant exists."""
        mgr = ModelCacheManager(tmp_path / "models")
        (tmp_path / "models" / "yolo" / "fp32").mkdir(parents=True)
        assert mgr.is_cached("yolo", "fp32") is True

    def test_with_variant_false(self, tmp_path: Path) -> None:
        """Returns False when the variant is missing."""
        mgr = ModelCacheManager(tmp_path / "models")
        assert mgr.is_cached("yolo", "fp32") is False

    def test_without_variant_true(self, tmp_path: Path) -> None:
        """Returns True when any variant of the model is present."""
        mgr = ModelCacheManager(tmp_path / "models")
        (tmp_path / "models" / "yolo").mkdir()
        assert mgr.is_cached("yolo") is True

    def test_without_variant_false(self, tmp_path: Path) -> None:
        """Returns False when no variant of the model is present."""
        mgr = ModelCacheManager(tmp_path / "models")
        assert mgr.is_cached("yolo") is False


@pytest.mark.unit
class TestListCachedModels:
    """Tests for ModelCacheManager.list_cached_models."""

    def test_returns_model_dirs(self, tmp_path: Path) -> None:
        """Returns the names of subdirectories under the cache dir."""
        mgr = ModelCacheManager(tmp_path / "models")
        (tmp_path / "models" / "yolo").mkdir()
        (tmp_path / "models" / "clip").mkdir()
        assert set(mgr.list_cached_models()) == {"yolo", "clip"}

    def test_ignores_files(self, tmp_path: Path) -> None:
        """Stray files at the cache root are not returned."""
        mgr = ModelCacheManager(tmp_path / "models")
        (tmp_path / "models" / "yolo").mkdir()
        (tmp_path / "models" / "stray.txt").write_text("x")
        assert mgr.list_cached_models() == ["yolo"]

    def test_returns_empty_when_root_missing(self, tmp_path: Path) -> None:
        """Returns [] when the cache dir does not exist."""
        mgr = ModelCacheManager(tmp_path / "models")
        (tmp_path / "models").rmdir()
        assert mgr.list_cached_models() == []


@pytest.mark.unit
class TestListCachedVariants:
    """Tests for ModelCacheManager.list_cached_variants."""

    def test_returns_variant_dirs(self, tmp_path: Path) -> None:
        """Returns the variant subdirectory names for a model."""
        mgr = ModelCacheManager(tmp_path / "models")
        (tmp_path / "models" / "yolo" / "fp32").mkdir(parents=True)
        (tmp_path / "models" / "yolo" / "int8").mkdir()
        assert set(mgr.list_cached_variants("yolo")) == {"fp32", "int8"}

    def test_ignores_files(self, tmp_path: Path) -> None:
        """Stray files inside the model dir are filtered out."""
        mgr = ModelCacheManager(tmp_path / "models")
        (tmp_path / "models" / "yolo" / "fp32").mkdir(parents=True)
        (tmp_path / "models" / "yolo" / "stray.txt").write_text("x")
        assert mgr.list_cached_variants("yolo") == ["fp32"]

    def test_returns_empty_when_model_missing(self, tmp_path: Path) -> None:
        """Returns [] when the model directory does not exist."""
        mgr = ModelCacheManager(tmp_path / "models")
        assert mgr.list_cached_variants("ghost") == []


@pytest.mark.unit
class TestListCacheContents:
    """Tests for ModelCacheManager.list_cache_contents."""

    def test_returns_mapping(self, tmp_path: Path) -> None:
        """Returns a model→variants mapping for every cached model."""
        mgr = ModelCacheManager(tmp_path / "models")
        (tmp_path / "models" / "yolo" / "fp32").mkdir(parents=True)
        (tmp_path / "models" / "yolo" / "int8").mkdir()
        (tmp_path / "models" / "clip" / "fp16").mkdir(parents=True)
        contents = mgr.list_cache_contents()
        assert set(contents) == {"yolo", "clip"}
        assert set(contents["yolo"]) == {"fp32", "int8"}
        assert contents["clip"] == ["fp16"]

    def test_returns_empty_when_root_missing(self, tmp_path: Path) -> None:
        """Returns {} when the cache dir is missing."""
        mgr = ModelCacheManager(tmp_path / "models")
        (tmp_path / "models").rmdir()
        assert mgr.list_cache_contents() == {}


@pytest.mark.unit
class TestRemoveVariant:
    """Tests for ModelCacheManager.remove_variant."""

    def test_removes_files_and_reports_size(self, tmp_path: Path) -> None:
        """Removing a variant deletes its files and returns total bytes freed."""
        mgr = ModelCacheManager(tmp_path / "models")
        variant_dir = tmp_path / "models" / "yolo" / "fp32"
        total = 0
        total += _write_file(variant_dir / "weights.bin", b"x" * 100)
        total += _write_file(variant_dir / "nested" / "config.json", b"y" * 25)

        freed = mgr.remove_variant("yolo", "fp32")

        assert freed == total
        assert not variant_dir.exists()

    def test_missing_raises_file_not_found(self, tmp_path: Path) -> None:
        """Removing a non-existent variant raises FileNotFoundError."""
        mgr = ModelCacheManager(tmp_path / "models")
        with pytest.raises(FileNotFoundError, match=r"Model variant directory .* does not exist"):
            mgr.remove_variant("yolo", "fp32")


@pytest.mark.unit
class TestRemoveModel:
    """Tests for ModelCacheManager.remove_model."""

    def test_removes_all_variants_and_reports_size(self, tmp_path: Path) -> None:
        """All variant subdirectories are removed and total size reported."""
        mgr = ModelCacheManager(tmp_path / "models")
        total = 0
        total += _write_file(tmp_path / "models" / "yolo" / "fp32" / "w.bin", b"a" * 10)
        total += _write_file(tmp_path / "models" / "yolo" / "int8" / "w.bin", b"b" * 7)

        freed = mgr.remove_model("yolo")

        assert freed == total
        assert not (tmp_path / "models" / "yolo").exists()

    def test_missing_raises_file_not_found(self, tmp_path: Path) -> None:
        """Removing a non-existent model raises FileNotFoundError."""
        mgr = ModelCacheManager(tmp_path / "models")
        with pytest.raises(FileNotFoundError, match=r"Model directory .* does not exist"):
            mgr.remove_model("ghost")

    def test_warns_and_removes_stray_file(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A stray file in the model dir is logged at WARNING and counted toward freed bytes."""
        mgr = ModelCacheManager(tmp_path / "models")
        stray_size = _write_file(tmp_path / "models" / "yolo" / "stray.txt", b"z" * 13)
        variant_size = _write_file(tmp_path / "models" / "yolo" / "fp32" / "w.bin", b"a" * 4)

        with caplog.at_level(logging.WARNING, logger="moment_to_action.paths._cache._models"):
            freed = mgr.remove_model("yolo")

        assert freed == stray_size + variant_size
        assert not (tmp_path / "models" / "yolo").exists()
        assert any(
            "stray.txt" in record.getMessage() and record.levelno == logging.WARNING
            for record in caplog.records
        )


@pytest.mark.unit
class TestClearCache:
    """Tests for ModelCacheManager.clear_cache."""

    def test_clears_all_models_and_removes_root(self, tmp_path: Path) -> None:
        """All models and the cache root are removed."""
        mgr = ModelCacheManager(tmp_path / "models")
        total = 0
        total += _write_file(tmp_path / "models" / "yolo" / "fp32" / "w.bin", b"a" * 11)
        total += _write_file(tmp_path / "models" / "clip" / "fp16" / "w.bin", b"b" * 22)

        freed = mgr.clear_cache()

        assert freed == total
        assert not (tmp_path / "models").exists()

    def test_clears_empty_cache(self, tmp_path: Path) -> None:
        """An empty cache yields 0 bytes freed and the root is removed."""
        mgr = ModelCacheManager(tmp_path / "models")
        freed = mgr.clear_cache()
        assert freed == 0
        assert not (tmp_path / "models").exists()
