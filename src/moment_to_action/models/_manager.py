from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import platformdirs

from ._registry import MODEL_REGISTRY
from ._types import (
    DownloadSource,
    ModelID,
    ModelInfo,
    ModelStatus,
    TransformersSource,
    VendoredSource,
)

if TYPE_CHECKING:
    from typing import IO

    import httpx


logger = logging.getLogger(__name__)


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

    def __init__(self, cache_dir: Path | None = None, *, show_progress: bool = True) -> None:
        """Initialize the model manager.

        Args:
            cache_dir: Override the default cache directory. If None, uses
                platformdirs.user_cache_path("moment_to_action", "GATech") / "models".
            show_progress: Show a Rich progress bar during model downloads.
                Set to False in tests or non-interactive environments.
        """
        if cache_dir is None:
            cache_dir = platformdirs.user_cache_path("moment_to_action", "GATech") / "models"

        self._cache_dir = cache_dir
        self._vendored_dir = Path(__file__).parent / "_vendored"
        self._show_progress = show_progress

        logger.info(
            "ModelManager initialized: cache_dir=%s, show_progress=%s",
            cache_dir,
            show_progress,
        )

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
        logger.debug("Resolving model path: %s", model.value)
        info = self._get_model_info(model)
        path = self._resolve_path(info)
        logger.debug("Model path resolved: %s → %s", model.value, path)
        return path

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
            available = path.exists()
            logger.debug("Model %s availability (vendored): %s", model.value, available)
            if available:
                return True
        except FileNotFoundError:
            pass

        # Vendored model not found; check cache
        if isinstance(info.source, DownloadSource):
            cache_path = self._cache_dir / model.value / info.filename
            available = cache_path.exists()
            logger.debug("Model %s availability (cached): %s", model.value, available)
            return available
        if isinstance(info.source, TransformersSource):
            cache_path = self._cache_dir / model.value
            available = self._is_transformers_cache_ready(cache_path)
            logger.debug("Model %s availability (cached): %s", model.value, available)
            return available
        logger.debug("Model %s not available", model.value)
        return False

    def list_models(self) -> list[ModelStatus]:
        """Return status of all known models without triggering downloads.

        Reports the current availability of each model: vendored models are
        always available if the package is intact; downloadable models are
        available only if already cached locally.

        Returns:
            List of ModelStatus for each model in the registry.
        """
        statuses = []
        for info in MODEL_REGISTRY.values():
            # Resolve the local path without triggering a download
            try:
                path = self._resolve_path_local(info)
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
        logger.info("Clearing model cache at: %s", self._cache_dir)
        removed_ids = []
        total_bytes = 0

        for model_id, info in MODEL_REGISTRY.items():
            if not isinstance(info.source, DownloadSource):
                continue  # Skip vendored models

            model_cache_dir = self._cache_dir / info.id.value
            if model_cache_dir.exists():
                bytes_freed = self._rmdir_with_size(model_cache_dir)
                total_bytes += bytes_freed
                removed_ids.append(model_id)
                logger.debug("Removed cached model %s (%d bytes)", model_id.value, bytes_freed)

        logger.info(
            "Cache cleared: removed %d models, %d bytes freed", len(removed_ids), total_bytes
        )
        return (total_bytes, removed_ids)

    @staticmethod
    def _rmdir_with_size(dir_: Path) -> int:
        """Remove a directory, returning its size.

        The directory should exist.
        """
        size = 0

        for item in dir_.glob("*"):
            if item.is_file():
                size += item.stat().st_size
                item.unlink()
            else:
                size += ModelManager._rmdir_with_size(item)

        dir_.rmdir()
        return size

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
                    logger.error("Vendored model not found: %s", path)
                    msg = f"Vendored model not found: {path}"
                    raise FileNotFoundError(msg)
                logger.debug("Using vendored model: %s", path)
                return path

            case DownloadSource(hf_repo_id=repo, hf_filename=hf_filename):
                cache_path = self._cache_dir / info.id.value / info.filename
                if cache_path.exists():
                    logger.debug("Using cached model: %s", cache_path)
                    return cache_path

                # Download from HuggingFace Hub
                logger.info("Downloading model %s from %s", info.id.value, repo)
                self._download_from_hf(repo, hf_filename, cache_path)
                logger.info("Model downloaded successfully: %s", cache_path)
                return cache_path

            case TransformersSource(hf_repo_id=repo):
                cache_dir = self._cache_dir / info.id.value
                if self._is_transformers_cache_ready(cache_dir):
                    logger.debug("Using cached transformers repo: %s", cache_dir)
                    return cache_dir

                logger.info("Downloading transformers model %s from %s", info.id.value, repo)
                self._download_transformers_model(repo, cache_dir)
                logger.info("Transformers model downloaded successfully: %s", cache_dir)
                return cache_dir

    def _resolve_path_local(self, info: ModelInfo) -> Path:
        """Resolve the local path for a model without triggering a download.

        For vendored models, returns the embedded path. For downloadable models,
        returns the cache path (whether or not the file exists yet).

        Args:
            info: The ModelInfo to resolve.

        Returns:
            Local path where the model file is (or would be).

        Raises:
            FileNotFoundError: If the model is vendored but missing from the
                package (corrupt/incomplete installation).
        """
        match info.source:
            case VendoredSource(subdir=subdir):
                path = self._vendored_dir / subdir / info.filename
                if not path.exists():
                    msg = f"Vendored model not found: {path}"
                    raise FileNotFoundError(msg)
                return path
            case DownloadSource():
                return self._cache_dir / info.id.value / info.filename
            case TransformersSource():
                return self._cache_dir / info.id.value

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
            case TransformersSource():
                msg = f"Model {info.id} is downloadable, not vendored"
                raise FileNotFoundError(msg)

    @staticmethod
    def _is_transformers_cache_ready(cache_dir: Path) -> bool:
        """Check whether a cached transformers repo directory looks usable."""
        if not cache_dir.exists() or not cache_dir.is_dir():
            return False

        has_model_config = (cache_dir / "config.json").exists()
        has_processor_config = (cache_dir / "preprocessor_config.json").exists() or (
            cache_dir / "tokenizer_config.json"
        ).exists()
        return has_model_config and has_processor_config

    def _download_from_hf(self, repo_id: str, filename: str, dest_path: Path) -> None:
        """Download a model from HuggingFace Hub with a Rich progress bar.

        Resolves the download URL and file size via the HuggingFace Hub API,
        then streams the file directly using httpx, updating a Rich progress
        bar as bytes arrive.

        Args:
            repo_id: HuggingFace repo ID (e.g.,
                "anton96vice/mobileclip2_tflite").
            filename: Filename in the repo.
            dest_path: Where to save the file.

        Raises:
            RuntimeError: If download fails or required dependencies are not
                installed.
        """
        try:
            import httpx
            from huggingface_hub import get_hf_file_metadata, hf_hub_url
        except ImportError as exc:
            msg = f"Required dependency not available: {exc.name}"
            raise RuntimeError(msg) from None

        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Downloading %s from %s to %s", filename, repo_id, dest_path)

            # Resolve the CDN download URL and fetch metadata (file size)
            url = hf_hub_url(repo_id=repo_id, filename=filename)
            metadata = get_hf_file_metadata(url)
            logger.debug("HF file size: %d bytes", metadata.size)

            with (
                dest_path.open("wb") as fh,
                httpx.stream("GET", metadata.location, follow_redirects=True) as resp,
            ):
                resp.raise_for_status()
                self._stream_with_progress(resp, fh, filename, metadata.size)

        except Exception as e:
            # Remove partial download so the next attempt starts fresh
            if dest_path.exists():
                dest_path.unlink()
            logger.exception("Download failed for %s/%s", repo_id, filename)
            msg = f"Failed to download {repo_id}/{filename}: {e}"
            raise RuntimeError(msg) from e

    def _download_transformers_model(self, repo_id: str, dest_dir: Path) -> None:
        """Download a transformers model repo and save it under a managed cache path.

        Args:
            repo_id: HuggingFace repo ID (e.g. "HuggingFaceTB/SmolVLM2-2.2B-Instruct").
            dest_dir: Target directory under the model cache.

        Raises:
            RuntimeError: If download fails or required dependencies are missing.
        """
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            msg = f"Required dependency not available: {exc.name}"
            raise RuntimeError(msg) from None

        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Downloading transformers repo %s to %s", repo_id, dest_dir)

            processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
            processor.save_pretrained(dest_dir)

            model = AutoModelForImageTextToText.from_pretrained(repo_id, trust_remote_code=True)
            model.save_pretrained(dest_dir)

        except Exception as e:
            if dest_dir.exists():
                self._rmdir_with_size(dest_dir)
            logger.exception("Transformers model download failed for %s", repo_id)
            msg = f"Failed to download transformers model {repo_id}: {e}"
            raise RuntimeError(msg) from e

    def _stream_with_progress(
        self,
        response: httpx.Response,
        dest: IO[bytes],
        description: str,
        total: int | None,
    ) -> None:
        """Write response bytes to dest, optionally showing a Rich progress bar.

        Args:
            response: The streaming httpx response to read from.
            dest: File-like object to write downloaded bytes into.
            description: Label shown next to the progress bar (e.g., filename).
            total: Expected total bytes, or None for indeterminate progress.
        """
        if self._show_progress:
            from rich.progress import (
                BarColumn,
                DownloadColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeRemainingColumn,
                TransferSpeedColumn,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(description, total=total)
                for chunk in response.iter_bytes(chunk_size=8192):
                    dest.write(chunk)
                    progress.update(task, advance=len(chunk))
            logger.debug(
                "Download completed with progress bar: %s (%d bytes)",
                description,
                total or 0,
            )
        else:
            bytes_written = 0
            for chunk in response.iter_bytes(chunk_size=8192):
                dest.write(chunk)
                bytes_written += len(chunk)
            logger.debug(
                "Download completed (silent mode): %s (%d bytes)",
                description,
                bytes_written,
            )
