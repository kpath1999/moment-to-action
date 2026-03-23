from __future__ import annotations

from pathlib import Path

import platformdirs

from ._registry import MODEL_REGISTRY
from ._types import (
    DownloadSource,
    ModelID,
    ModelInfo,
    ModelStatus,
    VendoredSource,
)


class ModelManager:
    """Manage ML model discovery, caching, and resolution.

    The ModelManager handles:
    - Discovery of vendored and downloadable models from MODEL_REGISTRY
    - Resolution of model file paths with automatic downloading from HuggingFace
    - Local caching of downloaded models
    - Model availability checks and cache management

    Models can be either vendored (included in the package under _vendored/)
    or downloadable from HuggingFace Hub. Downloaded models are cached locally
    to avoid re-downloading.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize the model manager.

        Args:
            cache_dir: Override the default cache directory. If None, uses
                platformdirs.user_cache_path("moment_to_action", "GATech") / "models".
        """
        if cache_dir is None:
            cache_dir = platformdirs.user_cache_path("moment_to_action", "GATech") / "models"

        self._cache_dir = cache_dir
        self._vendored_dir = Path(__file__).parent / "_vendored"

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory path.

        Returns:
            The directory where downloaded models are cached.
        """
        return self._cache_dir

    def get_path(self, model: ModelID) -> Path:
        """Return filesystem path for a model, downloading if needed.

        Args:
            model: The ModelID to get.

        Returns:
            Path to the model file.

        Raises:
            FileNotFoundError: If the model is not available and cannot be
                downloaded.
            RuntimeError: If model ID is not in registry.
        """
        info = self._get_model_info(model)
        return self._resolve_path(info)

    def is_available(self, model: ModelID) -> bool:
        """Check if model is available without downloading.

        Args:
            model: The ModelID to check.

        Returns:
            True if the model file exists locally.
        """
        info = self._get_model_info(model)
        try:
            path = self._resolve_path_vendored_only(info)
            return path.exists()
        except FileNotFoundError:
            # Vendored model not found; check cache
            if isinstance(info.source, DownloadSource):
                cache_path = self._cache_dir / model.value / info.filename
                return cache_path.exists()
            return False

    def list_models(self) -> list[ModelStatus]:
        """Return status of all known models.

        Returns:
            List of ModelStatus for each model in the registry.
        """
        statuses = []
        for info in MODEL_REGISTRY.values():
            try:
                path = self._resolve_path(info)
                available = path.exists()
                size = path.stat().st_size if available else None
            except FileNotFoundError:
                available = False
                path = None
                size = None

            statuses.append(
                ModelStatus(
                    info=info,
                    available=available,
                    path=path if available else None,
                    size_bytes=size,
                )
            )
        return statuses

    def clear_cache(self) -> tuple[int, list[ModelID]]:
        """Remove downloaded (non-vendored) models from cache.

        Returns:
            Tuple of (total_bytes_freed, list_of_model_ids_removed).
        """
        removed_ids = []
        total_bytes = 0

        for model_id, info in MODEL_REGISTRY.items():
            if not isinstance(info.source, DownloadSource):
                continue  # Skip vendored models

            model_cache_dir = self._cache_dir / model_id.value
            if model_cache_dir.exists():
                for item in model_cache_dir.rglob("*"):
                    if item.is_file():
                        total_bytes += item.stat().st_size
                        item.unlink()
                model_cache_dir.rmdir()
                removed_ids.append(model_id)

        return (total_bytes, removed_ids)

    def _get_model_info(self, model: ModelID) -> ModelInfo:
        """Get ModelInfo for a model ID.

        Raises:
            RuntimeError: If model ID not in registry.
        """
        if model not in MODEL_REGISTRY:
            msg = f"Unknown model: {model}"
            raise RuntimeError(msg)
        return MODEL_REGISTRY[model]

    def _resolve_path(self, info: ModelInfo) -> Path:
        """Resolve path, handling download if necessary.

        Uses match/case on the ModelSource union to dispatch vendored vs
        download logic.

        Args:
            info: The ModelInfo to resolve.

        Returns:
            Path to the model file.

        Raises:
            FileNotFoundError: If vendored model not found or download fails.
            RuntimeError: If huggingface_hub is not installed.
        """
        match info.source:
            case VendoredSource(subdir=subdir):
                path = self._vendored_dir / subdir / info.filename
                if not path.exists():
                    msg = f"Vendored model not found: {path}"
                    raise FileNotFoundError(msg)
                return path

            case DownloadSource(hf_repo_id=repo, hf_filename=hf_filename):
                cache_path = self._cache_dir / info.id.value / info.filename
                if cache_path.exists():
                    return cache_path

                # Download from HuggingFace Hub
                self._download_from_hf(repo, hf_filename, cache_path)
                return cache_path

    def _resolve_path_vendored_only(self, info: ModelInfo) -> Path:
        """Resolve path for vendored models only (no download).

        Args:
            info: The ModelInfo to resolve.

        Returns:
            Path to the vendored model file.

        Raises:
            FileNotFoundError: If the model is not vendored.
        """
        match info.source:
            case VendoredSource(subdir=subdir):
                return self._vendored_dir / subdir / info.filename
            case DownloadSource():
                msg = f"Model {info.id} is downloadable, not vendored"
                raise FileNotFoundError(msg)

    def _download_from_hf(self, repo_id: str, filename: str, dest_path: Path) -> None:
        """Download a model from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID (e.g.,
                "anton96vice/mobileclip2_tflite").
            filename: Filename in the repo.
            dest_path: Where to save the file.

        Raises:
            RuntimeError: If download fails or huggingface_hub is not
                installed.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            msg = "huggingface_hub not installed"
            raise RuntimeError(msg) from None

        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(dest_path.parent),
                local_files_only=False,
            )
            # Ensure the file is at the expected path
            if Path(downloaded_path) != dest_path:
                downloaded_path_obj = Path(downloaded_path)
                if downloaded_path_obj.exists():
                    downloaded_path_obj.rename(dest_path)
        except Exception as e:
            msg = f"Failed to download {repo_id}/{filename}: {e}"
            raise RuntimeError(msg) from e
