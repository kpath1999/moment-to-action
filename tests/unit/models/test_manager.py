"""Unit tests for ModelManager class."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from moment_to_action.models._manager import ModelManager
from moment_to_action.models._types import ModelID


@pytest.mark.unit
class TestModelManagerInit:
    """Tests for ModelManager.__init__."""

    def test_init_with_default_cache_dir(self) -> None:
        """Test __init__ with default cache directory using platformdirs."""
        manager = ModelManager()
        assert manager.cache_dir is not None
        assert isinstance(manager.cache_dir, Path)
        assert "moment_to_action" in str(manager.cache_dir).lower()

    def test_init_with_custom_cache_dir(self, tmp_path: Path) -> None:
        """Test __init__ with custom cache_dir override."""
        custom_cache = tmp_path / "custom_cache"
        manager = ModelManager(cache_dir=custom_cache)
        assert manager.cache_dir == custom_cache

    def test_init_sets_vendored_dir(self) -> None:
        """Test that __init__ sets _vendored_dir correctly."""
        manager = ModelManager()
        assert manager._vendored_dir is not None
        assert isinstance(manager._vendored_dir, Path)
        assert manager._vendored_dir.name == "_vendored"
        assert manager._vendored_dir.exists()

    def test_init_vendored_dir_points_to_models_package(self) -> None:
        """Test that _vendored_dir points to the models package."""
        manager = ModelManager()
        assert "models" in str(manager._vendored_dir)


@pytest.mark.unit
class TestModelManagerGetPath:
    """Tests for ModelManager.get_path()."""

    def test_get_path_for_yolo_v8_returns_path(self) -> None:
        """Test get_path() for YOLO_V8 returns correct path."""
        manager = ModelManager()
        path = manager.get_path(ModelID.YOLO_V8)
        assert isinstance(path, Path)
        assert path.name == "model.onnx"

    def test_get_path_for_yolo_v8_file_exists(self) -> None:
        """Test get_path() for YOLO_V8 returns path that exists."""
        manager = ModelManager()
        path = manager.get_path(ModelID.YOLO_V8)
        assert path.exists()

    def test_get_path_for_yolo_v8_is_file(self) -> None:
        """Test get_path() for YOLO_V8 returns path to a file."""
        manager = ModelManager()
        path = manager.get_path(ModelID.YOLO_V8)
        assert path.is_file()

    def test_get_path_for_missing_vendored_raises_file_not_found(self, tmp_path: Path) -> None:
        """Test get_path() raises FileNotFoundError for missing vendored model."""
        # Create a manager with custom vendored dir that doesn't have the model
        manager = ModelManager(cache_dir=tmp_path / "cache")
        manager._vendored_dir = tmp_path / "empty_vendored"
        manager._vendored_dir.mkdir(exist_ok=True)

        with pytest.raises(FileNotFoundError):
            manager.get_path(ModelID.YOLO_V8)

    def test_get_path_for_invalid_model_raises_runtime_error(self) -> None:
        """Test get_path() raises RuntimeError for unknown model ID."""
        manager = ModelManager()
        # Create a fake ModelID that's not in the registry by directly checking
        # that accessing an unknown ID in _get_model_info raises the error
        fake_id = ModelID.YOLO_V8
        # Temporarily empty the registry to test error handling
        from moment_to_action.models import _manager

        saved_registry = _manager.MODEL_REGISTRY.copy()
        try:
            _manager.MODEL_REGISTRY.clear()
            with pytest.raises(RuntimeError, match="Unknown model"):
                manager.get_path(fake_id)
        finally:
            _manager.MODEL_REGISTRY.update(saved_registry)

    @mock.patch("huggingface_hub.hf_hub_download")
    def test_get_path_for_downloadable_model_uses_cache_if_exists(
        self,
        mock_hf_download: mock.MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test get_path() returns cached model without downloading."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        manager = ModelManager(cache_dir=cache_dir)

        # Create a fake cached model
        model_cache = cache_dir / ModelID.MOBILECLIP_S2.value
        model_cache.mkdir(parents=True)
        cached_file = model_cache / "mobileclip_s2_datacompdr_last.tflite"
        cached_file.write_text("fake model content")

        path = manager.get_path(ModelID.MOBILECLIP_S2)
        assert path == cached_file
        # Should not have called download since cache exists
        mock_hf_download.assert_not_called()

    @mock.patch("huggingface_hub.hf_hub_download")
    def test_get_path_for_downloadable_model_downloads_if_not_cached(
        self,
        mock_hf_download: mock.MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test get_path() downloads model if not in cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        manager = ModelManager(cache_dir=cache_dir)

        # Create the expected cache location
        model_cache = cache_dir / ModelID.MOBILECLIP_S2.value
        model_cache.mkdir(parents=True)
        cached_file = model_cache / "mobileclip_s2_datacompdr_last.tflite"

        # Mock hf_hub_download to create the file
        def fake_download(*_args: object, **_kwargs: object) -> str:
            cached_file.write_text("fake model content")
            return str(cached_file)

        mock_hf_download.side_effect = fake_download

        path = manager.get_path(ModelID.MOBILECLIP_S2)
        assert path == cached_file
        # Should have called download
        mock_hf_download.assert_called_once()


@pytest.mark.unit
class TestModelManagerIsAvailable:
    """Tests for ModelManager.is_available()."""

    def test_is_available_for_yolo_v8_returns_true(self) -> None:
        """Test is_available() returns True for vendored YOLO_V8."""
        manager = ModelManager()
        assert manager.is_available(ModelID.YOLO_V8) is True

    def test_is_available_for_missing_vendored_returns_false(self, tmp_path: Path) -> None:
        """Test is_available() returns False for missing vendored model."""
        manager = ModelManager(cache_dir=tmp_path / "cache")
        manager._vendored_dir = tmp_path / "empty_vendored"
        manager._vendored_dir.mkdir(exist_ok=True)

        assert manager.is_available(ModelID.YOLO_V8) is False

    def test_is_available_for_non_cached_downloadable_returns_false(self, tmp_path: Path) -> None:
        """Test is_available() returns False for non-cached downloadable model."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        manager = ModelManager(cache_dir=cache_dir)

        assert manager.is_available(ModelID.MOBILECLIP_S2) is False

    def test_is_available_for_cached_downloadable_returns_true(self, tmp_path: Path) -> None:
        """Test is_available() returns True for cached downloadable model."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        manager = ModelManager(cache_dir=cache_dir)

        # Create a cached model
        model_cache = cache_dir / ModelID.MOBILECLIP_S2.value
        model_cache.mkdir(parents=True)
        cached_file = model_cache / "mobileclip_s2_datacompdr_last.tflite"
        cached_file.write_text("fake model content")

        assert manager.is_available(ModelID.MOBILECLIP_S2) is True


@pytest.mark.unit
class TestModelManagerListModels:
    """Tests for ModelManager.list_models()."""

    def test_list_models_returns_list(self) -> None:
        """Test list_models() returns a list."""
        manager = ModelManager()
        statuses = manager.list_models()
        assert isinstance(statuses, list)

    def test_list_models_contains_all_models(self) -> None:
        """Test list_models() returns all models in registry."""
        manager = ModelManager()
        statuses = manager.list_models()
        assert len(statuses) == 2

    def test_list_models_yolo_is_available(self) -> None:
        """Test list_models() shows YOLO_V8 as available."""
        manager = ModelManager()
        statuses = manager.list_models()
        yolo_status = next(s for s in statuses if s.info.id == ModelID.YOLO_V8)
        assert yolo_status.available is True
        assert yolo_status.path is not None
        assert yolo_status.path.exists()

    def test_list_models_yolo_has_size(self) -> None:
        """Test list_models() includes size for available YOLO_V8."""
        manager = ModelManager()
        statuses = manager.list_models()
        yolo_status = next(s for s in statuses if s.info.id == ModelID.YOLO_V8)
        assert yolo_status.size_bytes is not None
        assert yolo_status.size_bytes > 0

    @mock.patch("huggingface_hub.hf_hub_download")
    def test_list_models_mobileclip_not_available_by_default(
        self,
        mock_hf_download: mock.MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test list_models() shows MOBILECLIP_S2 as unavailable (not cached)."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        manager = ModelManager(cache_dir=cache_dir)

        # Mock hf_hub_download to return a path that doesn't exist
        # (simulating a failed download or missing file)
        mock_hf_download.return_value = str(tmp_path / "fake.tflite")

        statuses = manager.list_models()
        mobileclip_status = next(s for s in statuses if s.info.id == ModelID.MOBILECLIP_S2)
        # Since the returned path doesn't exist, available should be False
        assert mobileclip_status.available is False
        assert mobileclip_status.path is None
        assert mobileclip_status.size_bytes is None

    def test_list_models_mobileclip_available_when_cached(self, tmp_path: Path) -> None:
        """Test list_models() shows cached MOBILECLIP_S2 as available."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        manager = ModelManager(cache_dir=cache_dir)

        # Create a cached model
        model_cache = cache_dir / ModelID.MOBILECLIP_S2.value
        model_cache.mkdir(parents=True)
        cached_file = model_cache / "mobileclip_s2_datacompdr_last.tflite"
        cached_file.write_text("fake model content")

        statuses = manager.list_models()
        mobileclip_status = next(s for s in statuses if s.info.id == ModelID.MOBILECLIP_S2)
        assert mobileclip_status.available is True
        assert mobileclip_status.path == cached_file
        assert mobileclip_status.size_bytes == len("fake model content")

    def test_list_models_returns_model_status_instances(self) -> None:
        """Test list_models() returns ModelStatus instances."""
        manager = ModelManager()
        statuses = manager.list_models()
        from moment_to_action.models._types import ModelStatus

        for status in statuses:
            assert isinstance(status, ModelStatus)


@pytest.mark.unit
class TestModelManagerClearCache:
    """Tests for ModelManager.clear_cache()."""

    def test_clear_cache_empty_returns_zero(self, tmp_path: Path) -> None:
        """Test clear_cache() with empty cache returns (0, [])."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        manager = ModelManager(cache_dir=cache_dir)

        bytes_freed, removed_ids = manager.clear_cache()
        assert bytes_freed == 0
        assert removed_ids == []

    def test_clear_cache_removes_cached_model(self, tmp_path: Path) -> None:
        """Test clear_cache() successfully removes cached model."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        manager = ModelManager(cache_dir=cache_dir)

        # Create a cached model
        model_cache = cache_dir / ModelID.MOBILECLIP_S2.value
        model_cache.mkdir(parents=True)
        cached_file = model_cache / "mobileclip_s2_datacompdr_last.tflite"
        cached_file.write_text("fake model content")

        assert cached_file.exists()

        bytes_freed, removed_ids = manager.clear_cache()

        assert cached_file.parent.exists() is False
        assert ModelID.MOBILECLIP_S2 in removed_ids
        assert bytes_freed == len("fake model content")

    def test_clear_cache_does_not_remove_vendored(self, tmp_path: Path) -> None:
        """Test clear_cache() does not remove vendored models."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        manager = ModelManager(cache_dir=cache_dir)

        # clear_cache should only remove downloadable models
        _bytes_freed, removed_ids = manager.clear_cache()

        # YOLO_V8 is vendored, so it should not be in removed_ids
        assert ModelID.YOLO_V8 not in removed_ids
        # YOLO_V8 should still exist
        assert manager.get_path(ModelID.YOLO_V8).exists()

    def test_clear_cache_returns_bytes_freed(self, tmp_path: Path) -> None:
        """Test clear_cache() returns correct bytes freed."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        manager = ModelManager(cache_dir=cache_dir)

        # Create a cached model with known size
        model_cache = cache_dir / ModelID.MOBILECLIP_S2.value
        model_cache.mkdir(parents=True)
        cached_file = model_cache / "mobileclip_s2_datacompdr_last.tflite"
        test_content = "x" * 12345
        cached_file.write_text(test_content)

        bytes_freed, _ = manager.clear_cache()

        assert bytes_freed == 12345

    def test_clear_cache_with_multiple_files(self, tmp_path: Path) -> None:
        """Test clear_cache() with multiple files in cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        manager = ModelManager(cache_dir=cache_dir)

        # Create a cached model with multiple files
        model_cache = cache_dir / ModelID.MOBILECLIP_S2.value
        model_cache.mkdir(parents=True)
        file1 = model_cache / "mobileclip_s2_datacompdr_last.tflite"
        file2 = model_cache / "metadata.json"
        file1.write_text("content1")
        file2.write_text("content2")

        bytes_freed, removed_ids = manager.clear_cache()

        assert not model_cache.exists()
        assert ModelID.MOBILECLIP_S2 in removed_ids
        assert bytes_freed == len("content1") + len("content2")


@pytest.mark.unit
class TestModelManagerGetModelInfo:
    """Tests for ModelManager._get_model_info()."""

    def test_get_model_info_for_yolo_v8(self) -> None:
        """Test _get_model_info() for valid YOLO_V8."""
        manager = ModelManager()
        info = manager._get_model_info(ModelID.YOLO_V8)
        assert info.id == ModelID.YOLO_V8
        assert info.filename == "model.onnx"

    def test_get_model_info_for_mobileclip_s2(self) -> None:
        """Test _get_model_info() for valid MOBILECLIP_S2."""
        manager = ModelManager()
        info = manager._get_model_info(ModelID.MOBILECLIP_S2)
        assert info.id == ModelID.MOBILECLIP_S2
        assert info.filename == "mobileclip_s2_datacompdr_last.tflite"

    def test_get_model_info_for_unknown_model_raises_runtime_error(
        self,
    ) -> None:
        """Test _get_model_info() raises RuntimeError for unknown model."""
        manager = ModelManager()
        fake_id = ModelID.YOLO_V8
        from moment_to_action.models import _manager

        saved_registry = _manager.MODEL_REGISTRY.copy()
        try:
            _manager.MODEL_REGISTRY.clear()
            with pytest.raises(RuntimeError, match="Unknown model"):
                manager._get_model_info(fake_id)
        finally:
            _manager.MODEL_REGISTRY.update(saved_registry)


@pytest.mark.unit
class TestModelManagerResolvePath:
    """Tests for ModelManager._resolve_path()."""

    def test_resolve_path_for_vendored_model(self) -> None:
        """Test _resolve_path() for vendored model."""
        manager = ModelManager()
        info = manager._get_model_info(ModelID.YOLO_V8)
        path = manager._resolve_path(info)
        assert path.exists()
        assert path.name == "model.onnx"

    def test_resolve_path_uses_match_case(self) -> None:
        """Test _resolve_path() uses match/case on ModelSource union."""
        manager = ModelManager()
        from moment_to_action.models._types import (
            DownloadSource,
            VendoredSource,
        )

        # Test with VendoredSource
        vendored_info = manager._get_model_info(ModelID.YOLO_V8)
        assert isinstance(vendored_info.source, VendoredSource)
        path = manager._resolve_path(vendored_info)
        assert path.exists()

        # Test with DownloadSource (mocked)
        with mock.patch("huggingface_hub.hf_hub_download"):
            download_info = manager._get_model_info(ModelID.MOBILECLIP_S2)
            assert isinstance(download_info.source, DownloadSource)

    def test_resolve_path_vendored_missing_raises_file_not_found(self, tmp_path: Path) -> None:
        """Test _resolve_path() raises FileNotFoundError for missing vendored."""
        manager = ModelManager(cache_dir=tmp_path / "cache")
        manager._vendored_dir = tmp_path / "empty_vendored"
        manager._vendored_dir.mkdir(exist_ok=True)

        info = manager._get_model_info(ModelID.YOLO_V8)
        with pytest.raises(FileNotFoundError, match="Vendored model not found"):
            manager._resolve_path(info)


@pytest.mark.unit
class TestModelManagerResolvePathVendoredOnly:
    """Tests for ModelManager._resolve_path_vendored_only()."""

    def test_resolve_path_vendored_only_for_vendored_model(self) -> None:
        """Test _resolve_path_vendored_only() for vendored model."""
        manager = ModelManager()
        info = manager._get_model_info(ModelID.YOLO_V8)
        path = manager._resolve_path_vendored_only(info)
        assert isinstance(path, Path)
        assert path.name == "model.onnx"

    def test_resolve_path_vendored_only_for_download_raises_file_not_found(
        self,
    ) -> None:
        """Test _resolve_path_vendored_only() raises for downloadable model."""
        manager = ModelManager()
        info = manager._get_model_info(ModelID.MOBILECLIP_S2)
        with pytest.raises(FileNotFoundError, match="downloadable, not vendored"):
            manager._resolve_path_vendored_only(info)


@pytest.mark.unit
class TestModelManagerDownloadFromHF:
    """Tests for ModelManager._download_from_hf()."""

    @mock.patch("huggingface_hub.hf_hub_download")
    def test_download_from_hf_calls_hf_hub_download(
        self,
        mock_hf_download: mock.MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test _download_from_hf() calls hf_hub_download with correct args."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        manager = ModelManager(cache_dir=cache_dir)

        dest_path = cache_dir / "model.tflite"

        manager._download_from_hf(
            repo_id="user/repo",
            filename="model.tflite",
            dest_path=dest_path,
        )

        mock_hf_download.assert_called_once()
        call_kwargs = mock_hf_download.call_args[1]
        assert call_kwargs["repo_id"] == "user/repo"
        assert call_kwargs["filename"] == "model.tflite"

    @mock.patch("huggingface_hub.hf_hub_download")
    def test_download_from_hf_creates_parent_directory(
        self,
        tmp_path: Path,
    ) -> None:
        """Test _download_from_hf() creates parent directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        manager = ModelManager(cache_dir=cache_dir)

        dest_path = cache_dir / "subdir" / "model.tflite"

        manager._download_from_hf(
            repo_id="user/repo",
            filename="model.tflite",
            dest_path=dest_path,
        )

        assert dest_path.parent.exists()

    def test_download_from_hf_raises_runtime_error_if_hf_hub_not_installed(
        self,
        tmp_path: Path,
    ) -> None:
        """Test _download_from_hf() raises RuntimeError if huggingface_hub not available."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        manager = ModelManager(cache_dir=cache_dir)

        dest_path = cache_dir / "model.tflite"

        # Mock the import to raise ImportError
        with mock.patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'huggingface_hub'"),
        ):
            with pytest.raises(RuntimeError, match="huggingface_hub not installed"):
                manager._download_from_hf(
                    repo_id="user/repo",
                    filename="model.tflite",
                    dest_path=dest_path,
                )

    @mock.patch("huggingface_hub.hf_hub_download")
    def test_download_from_hf_renames_if_path_differs(
        self,
        mock_hf_download: mock.MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test _download_from_hf() renames file if hf_hub_download returns different path."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        manager = ModelManager(cache_dir=cache_dir)

        dest_path = cache_dir / "model.tflite"
        other_path = cache_dir / "other.tflite"

        # Create the "downloaded" file at the other path
        other_path.write_text("test content")

        # Mock to return the other path
        mock_hf_download.return_value = str(other_path)

        manager._download_from_hf(
            repo_id="user/repo",
            filename="model.tflite",
            dest_path=dest_path,
        )

        # File should be at dest_path now
        assert dest_path.exists()
        assert dest_path.read_text() == "test content"
        assert not other_path.exists()

    @mock.patch("huggingface_hub.hf_hub_download")
    def test_download_from_hf_raises_runtime_error_on_download_failure(
        self,
        mock_hf_download: mock.MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test _download_from_hf() raises RuntimeError on download failure."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        manager = ModelManager(cache_dir=cache_dir)

        mock_hf_download.side_effect = Exception("Download failed")

        dest_path = cache_dir / "model.tflite"

        with pytest.raises(RuntimeError, match="Failed to download"):
            manager._download_from_hf(
                repo_id="user/repo",
                filename="model.tflite",
                dest_path=dest_path,
            )

    def test_is_available_returns_false_for_missing_vendored_non_download(
        self,
        tmp_path: Path,
    ) -> None:
        """Test is_available returns False when vendored model missing and not downloadable."""
        manager = ModelManager(cache_dir=tmp_path)

        with mock.patch.object(manager, "_resolve_path_vendored_only") as mock_resolve:
            mock_resolve.side_effect = FileNotFoundError("Not found")

            with mock.patch.object(manager, "_get_model_info") as mock_info:
                from moment_to_action.models._types import ModelInfo, VendoredSource

                mock_info.return_value = ModelInfo(
                    id=ModelID.YOLO_V8,
                    filename="test.onnx",
                    source=VendoredSource(subdir="test"),
                )

                result = manager.is_available(ModelID.YOLO_V8)
                assert result is False

    def test_list_models_exception_on_resolve_path(self, tmp_path: Path) -> None:
        """Test list_models handles FileNotFoundError when resolving paths."""
        manager = ModelManager(cache_dir=tmp_path)

        with mock.patch.object(manager, "_resolve_path") as mock_resolve:
            mock_resolve.side_effect = FileNotFoundError("Model not found")

            statuses = manager.list_models()

            assert len(statuses) == 2
            for status in statuses:
                assert not status.available
                assert status.path is None
                assert status.size_bytes is None
