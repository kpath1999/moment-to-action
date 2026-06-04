"""Unit tests for ModelManager."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from moment_to_action.models import (
    DEFAULT_VARIANT_KEY,
    DownloadSource,
    ModelFormat,
    ModelID,
    ModelInfo,
    ModelManager,
    YOLOModel,
)

if TYPE_CHECKING:
    from moment_to_action.paths import PathManager


def _custom_registry(info: ModelInfo) -> dict[ModelID, ModelInfo]:
    """Wrap one ModelInfo in a registry dict for ModelManager construction."""
    return {info.id: info}


def _download_info() -> ModelInfo:
    """Build a single-variant DownloadSource ModelInfo for MOBILECLIP_S2."""
    return ModelInfo(
        id=ModelID.MOBILECLIP_S2,
        model_class=YOLOModel,
        variants={
            DEFAULT_VARIANT_KEY: DownloadSource(
                format=ModelFormat.ONNX,
                url="https://example.com/model.bin",
                filename="model.bin",
            ),
        },
    )


@pytest.mark.unit
class TestInit:
    """Tests for ModelManager.__init__."""

    def test_stores_path_manager(self, path_manager: PathManager) -> None:
        """The supplied PathManager is retained."""
        mgr = ModelManager(path_manager)
        assert mgr._path_manager is path_manager

    def test_show_progress_defaults_to_true(self, path_manager: PathManager) -> None:
        """`show_progress` defaults to True."""
        mgr = ModelManager(path_manager)
        assert mgr._show_progress is True

    def test_show_progress_false(self, path_manager: PathManager) -> None:
        """`show_progress=False` is stored."""
        mgr = ModelManager(path_manager, show_progress=False)
        assert mgr._show_progress is False

    def test_custom_registry_is_used(self, path_manager: PathManager) -> None:
        """A custom `registry` overrides the default."""
        info = _download_info()
        mgr = ModelManager(path_manager, registry=_custom_registry(info))
        assert mgr._registry == {info.id: info}


@pytest.mark.unit
class TestGetModelInfo:
    """Tests for `_get_model_info`."""

    def test_returns_info_for_registered_model(self, path_manager: PathManager) -> None:
        """Returns ModelInfo for a registered model."""
        info = _download_info()
        mgr = ModelManager(path_manager, registry=_custom_registry(info))
        assert mgr._get_model_info(ModelID.MOBILECLIP_S2) is info

    def test_raises_value_error_for_unknown(self, path_manager: PathManager) -> None:
        """Raises ValueError when the model is not in the registry."""
        mgr = ModelManager(path_manager, registry={})
        with pytest.raises(ValueError, match="not found in registry"):
            mgr._get_model_info(ModelID.YOLO_V8)


@pytest.mark.unit
class TestGetSource:
    """Tests for `_get_source`."""

    def test_returns_source_for_variant(self) -> None:
        """Returns the source bound to the variant key."""
        info = _download_info()
        src = ModelManager._get_source(info, DEFAULT_VARIANT_KEY)
        assert src is info.variants[DEFAULT_VARIANT_KEY]

    def test_raises_value_error_for_unknown_variant(self) -> None:
        """Raises ValueError for a missing variant."""
        info = _download_info()
        with pytest.raises(ValueError, match="Variant 'ghost' not found"):
            ModelManager._get_source(info, "ghost")


@pytest.mark.unit
class TestAvailable:
    """Tests for `_available`."""

    def test_none_returns_false(self) -> None:
        """`_available(None)` is False."""
        assert ModelManager._available(None) is False

    def test_existing_path_returns_true(self, tmp_path: Path) -> None:
        """An existing path returns True."""
        f = tmp_path / "a"
        f.write_text("x")
        assert ModelManager._available(f) is True

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        """A non-existent path raises RuntimeError (invariant violation)."""
        with pytest.raises(RuntimeError, match="does not exist"):
            ModelManager._available(tmp_path / "ghost")


@pytest.mark.unit
class TestGetModelCacheDir:
    """Tests for `_get_model_cache_dir`."""

    def test_returns_variant_dir(self, path_manager: PathManager) -> None:
        """Returns the cache models manager's variant dir for `(model_id, variant)`."""
        mgr = ModelManager(path_manager)
        path = mgr._get_model_cache_dir(ModelID.MOBILECLIP_S2, DEFAULT_VARIANT_KEY)
        expected = path_manager.cache.models.models_dir / "mobileclip_s2" / DEFAULT_VARIANT_KEY
        assert path == expected


@pytest.mark.unit
class TestResolveModel:
    """Tests for `_resolve_model`."""

    def test_resolve_with_download_creates_variant_dir(
        self,
        path_manager: PathManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When download=True, the variant dir is created before resolving the source."""
        info = _download_info()
        mgr = ModelManager(path_manager, registry=_custom_registry(info), show_progress=False)

        captured: dict[str, object] = {}

        def fake_resolve(
            _source: object,
            variant_dir: Path,
            *,
            download: bool,
            progress: bool,
        ) -> None:
            captured["dir"] = variant_dir
            captured["download"] = download
            captured["progress"] = progress

        monkeypatch.setattr("moment_to_action.models._manager.resolve_model_source", fake_resolve)

        mgr._resolve_model(ModelID.MOBILECLIP_S2, DEFAULT_VARIANT_KEY, download=True)

        captured_dir = captured["dir"]
        assert isinstance(captured_dir, Path)
        assert captured_dir.exists()
        assert captured["download"] is True
        assert captured["progress"] is False

    def test_resolve_without_download_does_not_create_dir(
        self,
        path_manager: PathManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When download=False, the variant dir is not created proactively."""
        info = _download_info()
        mgr = ModelManager(path_manager, registry=_custom_registry(info))

        captured: dict[str, Path] = {}

        def fake_resolve(
            _source: object,
            variant_dir: Path,
            *,
            download: bool,  # noqa: ARG001
            progress: bool,  # noqa: ARG001
        ) -> None:
            captured["dir"] = variant_dir

        monkeypatch.setattr("moment_to_action.models._manager.resolve_model_source", fake_resolve)

        mgr._resolve_model(ModelID.MOBILECLIP_S2, DEFAULT_VARIANT_KEY, download=False)

        assert not captured["dir"].exists()


@pytest.mark.unit
class TestGetPath:
    """Tests for `get_path`."""

    def test_returns_path_when_resolved(
        self,
        path_manager: PathManager,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Returns the resolved path when the source provides one."""
        info = _download_info()
        mgr = ModelManager(path_manager, registry=_custom_registry(info), show_progress=False)
        target = tmp_path / "blob.bin"
        target.write_text("hi")

        monkeypatch.setattr(
            "moment_to_action.models._manager.resolve_model_source",
            lambda *_a, **_kw: target,
        )

        assert mgr.get_path(ModelID.MOBILECLIP_S2) == target

    def test_raises_when_source_returns_none(
        self,
        path_manager: PathManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Raises RuntimeError if `_resolve_model` returns None after download attempt."""
        info = _download_info()
        mgr = ModelManager(path_manager, registry=_custom_registry(info), show_progress=False)

        monkeypatch.setattr(
            "moment_to_action.models._manager.resolve_model_source",
            lambda *_a, **_kw: None,
        )

        with pytest.raises(RuntimeError, match="Download succeeded but model file not found"):
            mgr.get_path(ModelID.MOBILECLIP_S2)


@pytest.mark.unit
class TestIsAvailable:
    """Tests for `is_available`."""

    def test_returns_true_when_resolved(
        self,
        path_manager: PathManager,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Returns True when the variant resolves to an existing path."""
        info = _download_info()
        mgr = ModelManager(path_manager, registry=_custom_registry(info))
        target = tmp_path / "blob.bin"
        target.write_text("x")

        monkeypatch.setattr(
            "moment_to_action.models._manager.resolve_model_source",
            lambda *_a, **_kw: target,
        )

        assert mgr.is_available(ModelID.MOBILECLIP_S2) is True

    def test_returns_false_when_unresolved(
        self,
        path_manager: PathManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Returns False when the resolver yields None (no download attempted)."""
        info = _download_info()
        mgr = ModelManager(path_manager, registry=_custom_registry(info))

        monkeypatch.setattr(
            "moment_to_action.models._manager.resolve_model_source",
            lambda *_a, **_kw: None,
        )

        assert mgr.is_available(ModelID.MOBILECLIP_S2) is False


@pytest.mark.unit
class TestListModels:
    """Tests for `list_models`."""

    def test_lists_all_registered_models(self, path_manager: PathManager) -> None:
        """One ModelStatus per registry entry, with a VariantStatus per variant."""
        info = _download_info()
        mgr = ModelManager(path_manager, registry=_custom_registry(info))

        with mock.patch(
            "moment_to_action.models._manager.resolve_model_source",
            return_value=None,
        ):
            statuses = mgr.list_models()

        assert len(statuses) == 1
        s = statuses[0]
        assert s.info is info
        assert len(s.variants) == 1
        assert s.variants[0].available is False
        assert s.variants[0].size_bytes is None
        assert s.path is None
        assert s.available is False

    def test_size_and_path_set_for_available_variant(
        self,
        path_manager: PathManager,
        tmp_path: Path,
    ) -> None:
        """For an available variant, `size_bytes` reflects on-disk size and model path is set."""
        info = _download_info()
        mgr = ModelManager(path_manager, registry=_custom_registry(info))

        target = tmp_path / "blob.bin"
        target.write_bytes(b"x" * 123)

        with mock.patch(
            "moment_to_action.models._manager.resolve_model_source",
            return_value=target,
        ):
            statuses = mgr.list_models()

        s = statuses[0]
        assert s.available is True
        assert s.variants[0].size_bytes == 123
        assert s.path is not None
        assert s.path.name == info.id.value


@pytest.mark.unit
class TestClearCache:
    """Tests for `clear_cache`."""

    def test_delegates_to_models_cache_clear(self, path_manager: PathManager) -> None:
        """`clear_cache` returns the size reported by the model cache manager."""
        mgr = ModelManager(path_manager)
        # Drop a fake cached file under the model cache.
        target = path_manager.cache.models.models_dir / "yolo_v8" / DEFAULT_VARIANT_KEY / "blob.bin"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"y" * 50)

        contents = mgr.clear_cache()
        # ModelManager.clear_cache returns ModelCacheContents (the inner result).
        assert contents.total_size_bytes == 50
        assert not (path_manager.cache.models.models_dir).exists()


@pytest.mark.unit
class TestRemoveVariant:
    """Tests for remove_variant()."""

    def test_delegates_to_cache_remove_variant(self, path_manager: PathManager) -> None:
        """remove_variant delegates to path_manager.cache.models.remove_variant."""
        mgr = ModelManager(path_manager)
        target = path_manager.cache.models.models_dir / "yolo_v8" / DEFAULT_VARIANT_KEY / "blob.bin"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 64)

        freed = mgr.remove_variant(ModelID.YOLO_V8, DEFAULT_VARIANT_KEY)
        assert freed == 64
        assert not target.exists()

    def test_raises_when_variant_not_cached(self, path_manager: PathManager) -> None:
        """remove_variant raises FileNotFoundError if the variant directory is absent."""
        mgr = ModelManager(path_manager)
        with pytest.raises(FileNotFoundError):
            mgr.remove_variant(ModelID.YOLO_V8, "nonexistent_variant")


@pytest.mark.unit
class TestRemoveModel:
    """Tests for remove_model()."""

    def test_delegates_to_cache_remove_model(self, path_manager: PathManager) -> None:
        """remove_model delegates to path_manager.cache.models.remove_model."""
        mgr = ModelManager(path_manager)
        target = path_manager.cache.models.models_dir / "yolo_v8" / DEFAULT_VARIANT_KEY / "blob.bin"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"y" * 128)

        info = mgr.remove_model(ModelID.YOLO_V8)
        assert info.size_bytes == 128
        assert info.model_id == "yolo_v8"
        assert not (path_manager.cache.models.models_dir / "yolo_v8").exists()

    def test_raises_when_model_not_cached(self, path_manager: PathManager) -> None:
        """remove_model raises FileNotFoundError if the model directory is absent."""
        mgr = ModelManager(path_manager)
        with pytest.raises(FileNotFoundError):
            mgr.remove_model(ModelID.MOBILECLIP_S2)


@pytest.mark.unit
class TestGetModel:
    """Tests for get_model()."""

    def test_returns_yolo_model_instance(self, path_manager: PathManager) -> None:
        """get_model() returns a YOLOModel for YOLO_V8."""
        mgr = ModelManager(path_manager)
        model = mgr.get_model(ModelID.YOLO_V8)
        assert isinstance(model, YOLOModel)

    def test_model_not_loaded(self, path_manager: PathManager) -> None:
        """Returned model is unloaded (_backend is None)."""
        mgr = ModelManager(path_manager)
        model = mgr.get_model(ModelID.YOLO_V8)
        assert model._backend is None

    def test_model_variant_is_default(self, path_manager: PathManager) -> None:
        """get_model() sets _variant to the requested variant key."""
        mgr = ModelManager(path_manager)
        model = mgr.get_model(ModelID.YOLO_V8)
        assert model._variant == DEFAULT_VARIANT_KEY

    def test_model_format_matches_source(self, path_manager: PathManager) -> None:
        """get_model() passes source.format to the model constructor."""
        mgr = ModelManager(path_manager)
        model = mgr.get_model(ModelID.YOLO_V8)
        assert isinstance(model, YOLOModel)
        assert model._format == ModelFormat.ONNX

    def test_model_path_exists(self, path_manager: PathManager) -> None:
        """get_model() resolves path to an existing file."""
        mgr = ModelManager(path_manager)
        model = mgr.get_model(ModelID.YOLO_V8)
        assert model._path.exists()

    def test_model_kwargs_forwarded_to_constructor(self, path_manager: PathManager) -> None:
        """model_kwargs are forwarded verbatim to the model constructor."""
        import pytest

        mgr = ModelManager(path_manager)
        model = mgr.get_model(ModelID.YOLO_V8, confidence_threshold=0.3)
        assert isinstance(model, YOLOModel)
        assert model.confidence_threshold == pytest.approx(0.3)


@pytest.mark.unit
class TestVendoredFlow:
    """End-to-end-ish test using the real vendored YOLO model."""

    def test_yolo_v8_default_is_available(self, path_manager: PathManager) -> None:
        """Vendored YOLO_V8 default variant is available via the default registry."""
        mgr = ModelManager(path_manager)
        assert mgr.is_available(ModelID.YOLO_V8) is True

    def test_yolo_v8_get_path_returns_existing_file(self, path_manager: PathManager) -> None:
        """`get_path` returns the on-disk vendored YOLO file."""
        mgr = ModelManager(path_manager)
        p = mgr.get_path(ModelID.YOLO_V8)
        assert p.exists()
        assert p.name == "model.onnx"
