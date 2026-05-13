"""Unit tests for moment_to_action.paths._cache._models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from moment_to_action.paths._cache._models import (
    CachedModelInfo,
    ModelCacheContents,
    ModelCacheManager,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_file(path: Path, payload: bytes) -> int:
    """Create `path` (and parents) with the given bytes; return the byte count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return len(payload)


# ---------------------------------------------------------------------------
# CachedModelInfo
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCachedModelInfo:
    """Tests for CachedModelInfo."""

    def test_to_json_serializes_paths(self, tmp_path: Path) -> None:
        """to_json converts Path entries in `other` to strings."""
        stray = tmp_path / "stray.txt"
        info = CachedModelInfo(
            model_id="yolo",
            size_bytes=42,
            variants=["fp32"],
            other=[stray],
        )
        as_json = info.to_json()
        assert as_json["model_id"] == "yolo"
        assert as_json["size_bytes"] == 42
        assert as_json["variants"] == ["fp32"]
        assert as_json["other"] == [str(stray)]

    def test_to_rich_table_row_clean(self) -> None:
        """Row reports zero dirty files in green when other is empty."""
        info = CachedModelInfo(
            model_id="yolo",
            size_bytes=1024,
            variants=["fp32", "int8"],
            other=[],
        )
        row = info.to_rich_table_row()
        variants_cell = str(row[2])
        assert row[0] == "yolo"
        assert "1 KiB" in str(row[1])
        assert "fp32" in variants_cell
        assert "int8" in variants_cell
        assert row[3] == "[green]0[/green]"

    def test_to_rich_table_row_dirty(self, tmp_path: Path) -> None:
        """Row reports non-zero dirty count in red when other is non-empty."""
        info = CachedModelInfo(
            model_id="yolo",
            size_bytes=0,
            variants=[],
            other=[tmp_path / "a", tmp_path / "b"],
        )
        row = info.to_rich_table_row()
        assert row[3] == "[red]2[/red]"


# ---------------------------------------------------------------------------
# ModelCacheContents
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelCacheContents:
    """Tests for ModelCacheContents."""

    def test_item_count_returns_number_of_models(self) -> None:
        """item_count is the number of model entries."""
        contents = ModelCacheContents(
            total_size_bytes=0,
            models={
                "yolo": CachedModelInfo("yolo", 0, [], []),
                "clip": CachedModelInfo("clip", 0, [], []),
            },
            other=[],
        )
        assert contents.item_count == 2

    def test_to_json_serializes_models_and_other(self, tmp_path: Path) -> None:
        """to_json includes total size, model dict, and string `other` paths."""
        stray = tmp_path / "stray"
        contents = ModelCacheContents(
            total_size_bytes=100,
            models={"yolo": CachedModelInfo("yolo", 100, ["fp32"], [])},
            other=[stray],
        )
        as_json = contents.to_json()
        models_dict = as_json["models"]
        assert as_json["total_size_bytes"] == 100
        assert isinstance(models_dict, dict)
        assert models_dict["yolo"]["model_id"] == "yolo"
        assert as_json["other"] == [str(stray)]

    def test_models_to_rich_table_lists_models(self) -> None:
        """The rich table contains one row per model."""
        contents = ModelCacheContents(
            total_size_bytes=0,
            models={
                "yolo": CachedModelInfo("yolo", 10, ["fp32"], []),
                "clip": CachedModelInfo("clip", 20, ["fp16"], []),
            },
            other=[],
        )
        table = contents.models_to_rich_table()
        assert table.row_count == 2

    def test_models_to_rich_table_appends_other_section(self, tmp_path: Path) -> None:
        """When `other` is non-empty, an extra row is appended to the table."""
        contents = ModelCacheContents(
            total_size_bytes=0,
            models={"yolo": CachedModelInfo("yolo", 10, ["fp32"], [])},
            other=[tmp_path / "stray.bin"],
        )
        table = contents.models_to_rich_table()
        assert table.row_count == 2  # model row + footer row


# ---------------------------------------------------------------------------
# ModelCacheManager
# ---------------------------------------------------------------------------


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
        ModelCacheManager(target)
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

    def test_returns_path_under_models(self, tmp_path: Path) -> None:
        """get_model_dir returns `<models>/<model_id>` without creating it."""
        mgr = ModelCacheManager(tmp_path / "models")
        path = mgr.get_model_dir("yolo")
        assert path == tmp_path / "models" / "yolo"
        assert not path.exists()


@pytest.mark.unit
class TestGetVariantDir:
    """Tests for ModelCacheManager.get_variant_dir."""

    def test_returns_path_under_model(self, tmp_path: Path) -> None:
        """get_variant_dir returns `<models>/<model_id>/<variant>` without creating it."""
        mgr = ModelCacheManager(tmp_path / "models")
        path = mgr.get_variant_dir("yolo", "fp32")
        assert path == tmp_path / "models" / "yolo" / "fp32"
        assert not path.exists()


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
class TestListCachedModels:
    """Tests for ModelCacheManager.list_cached_models."""

    def test_returns_info_per_model(self, tmp_path: Path) -> None:
        """Returns CachedModelInfo entries keyed by model name."""
        mgr = ModelCacheManager(tmp_path / "models")
        size_yolo = _write_file(tmp_path / "models" / "yolo" / "fp32" / "w.bin", b"a" * 10)
        size_clip = _write_file(tmp_path / "models" / "clip" / "fp16" / "w.bin", b"b" * 17)

        out = mgr.list_cached_models()
        assert set(out) == {"yolo", "clip"}
        assert out["yolo"].variants == ["fp32"]
        assert out["yolo"].size_bytes == size_yolo
        assert out["clip"].size_bytes == size_clip

    def test_returns_empty_when_root_missing(self, tmp_path: Path) -> None:
        """Returns {} when the cache dir does not exist."""
        mgr = ModelCacheManager(tmp_path / "models")
        (tmp_path / "models").rmdir()
        assert mgr.list_cached_models() == {}

    def test_collects_other_when_unexpected_file_in_model_dir(self, tmp_path: Path) -> None:
        """Files at the model-dir level appear in `other`, not in variants."""
        mgr = ModelCacheManager(tmp_path / "models")
        _write_file(tmp_path / "models" / "yolo" / "fp32" / "w.bin", b"a" * 5)
        stray = tmp_path / "models" / "yolo" / "note.txt"
        stray.write_text("hi")

        info = mgr.list_cached_models()["yolo"]
        assert info.variants == ["fp32"]
        assert stray in info.other


@pytest.mark.unit
class TestListCacheContents:
    """Tests for ModelCacheManager.list_cache_contents."""

    def test_aggregates_size_from_model_size_bytes(self, tmp_path: Path) -> None:
        """Total size uses already-computed model size_bytes (no re-walk).

        Verifies the additive shortcut: total_size == sum(model.size_bytes) + sum(other files).
        """
        mgr = ModelCacheManager(tmp_path / "models")
        size_yolo = _write_file(tmp_path / "models" / "yolo" / "fp32" / "w.bin", b"a" * 11)
        size_clip = _write_file(tmp_path / "models" / "clip" / "fp16" / "w.bin", b"b" * 22)

        contents = mgr.list_cache_contents()

        assert contents.total_size_bytes == size_yolo + size_clip
        assert set(contents.models) == {"yolo", "clip"}
        assert contents.other == []

    def test_includes_stray_root_files_in_other_and_size(self, tmp_path: Path) -> None:
        """Stray files at the model-cache root are in `other` and counted toward size."""
        mgr = ModelCacheManager(tmp_path / "models")
        stray_size = _write_file(tmp_path / "models" / "stray.txt", b"x" * 7)
        model_size = _write_file(tmp_path / "models" / "yolo" / "fp32" / "w.bin", b"a" * 13)

        contents = mgr.list_cache_contents()

        assert contents.total_size_bytes == stray_size + model_size
        assert (tmp_path / "models" / "stray.txt") in contents.other

    def test_empty_cache(self, tmp_path: Path) -> None:
        """An empty cache yields zero size, no models, no other."""
        mgr = ModelCacheManager(tmp_path / "models")
        contents = mgr.list_cache_contents()
        assert contents.total_size_bytes == 0
        assert contents.models == {}
        assert contents.other == []


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

        info = mgr.remove_model("yolo")

        assert info.model_id == "yolo"
        assert info.size_bytes == total
        assert set(info.variants) == {"fp32", "int8"}
        assert info.other == []
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
            info = mgr.remove_model("yolo")

        assert info.size_bytes == stray_size + variant_size
        assert info.variants == ["fp32"]
        assert len(info.other) == 1
        assert info.other[0].name == "stray.txt"
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

        contents = mgr.clear_cache()

        assert contents.total_size_bytes == total
        assert set(contents.models) == {"yolo", "clip"}
        assert contents.other == []
        assert not (tmp_path / "models").exists()

    def test_clears_empty_cache(self, tmp_path: Path) -> None:
        """An empty cache yields 0 bytes freed and the root is removed."""
        mgr = ModelCacheManager(tmp_path / "models")
        contents = mgr.clear_cache()
        assert contents.total_size_bytes == 0
        assert contents.models == {}
        assert contents.other == []
        assert not (tmp_path / "models").exists()

    def test_clears_unexpected_root_file(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A stray file at the cache root is unlinked, counted, and a warning is logged."""
        mgr = ModelCacheManager(tmp_path / "models")
        stray_size = _write_file(tmp_path / "models" / "stray.txt", b"x" * 9)

        with caplog.at_level(logging.WARNING, logger="moment_to_action.paths._cache._models"):
            contents = mgr.clear_cache()

        assert contents.total_size_bytes == stray_size
        assert len(contents.other) == 1
        assert contents.other[0].name == "stray.txt"
        assert not (tmp_path / "models").exists()
        assert any(
            "stray.txt" in record.getMessage() and record.levelno == logging.WARNING
            for record in caplog.records
        )

    def test_clears_empty_dirs_treated_as_models(self, tmp_path: Path) -> None:
        """Empty subdirs at the root are treated as (empty) models and removed."""
        mgr = ModelCacheManager(tmp_path / "models")
        (tmp_path / "models" / "empty_model").mkdir()

        contents = mgr.clear_cache()

        assert contents.total_size_bytes == 0
        assert "empty_model" in contents.models
        assert not (tmp_path / "models").exists()

    def test_clears_residual_root_dir_in_second_pass(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A root-level dir that escapes `_cached_model_paths` is still cleared.

        The second loop in `clear_cache` defends against stray directories by walking
        `iterdir` after model removal. Force the first pass to skip it by monkeypatching
        `_cached_model_paths` to return empty.
        """
        mgr = ModelCacheManager(tmp_path / "models")
        size = _write_file(tmp_path / "models" / "ghost" / "blob.bin", b"x" * 13)

        monkeypatch.setattr(mgr, "_cached_model_paths", list)

        with caplog.at_level(logging.WARNING, logger="moment_to_action.paths._cache._models"):
            contents = mgr.clear_cache()

        assert contents.total_size_bytes == size
        assert any(p.name == "ghost" for p in contents.other)
        assert not (tmp_path / "models").exists()
