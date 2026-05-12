import logging
from pathlib import Path
from typing import overload

from moment_to_action.utils.files import clear_directory

log = logging.getLogger(__name__)


class ModelCacheManager:
    """Manager for handling cached models."""

    def __init__(self, model_cache_dir: Path) -> None:
        """Initialize the model cache manager with the given cache directory.

        Args:
            model_cache_dir: The directory where cached models will be stored.
        """
        self._dir = model_cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def models_dir(self) -> Path:
        """Return the directory for cached models."""
        return self._dir

    def _model_path(self, model_id: str, variant: str | None = None) -> Path:
        """Return the path for a specific model variant.

        Args:
            model_id: The ID of the model.
            variant: The variant of the model.

        Returns:
            The path to the directory for the specified model variant.
        """
        if variant is not None:
            return self._dir / model_id / variant
        return self._dir / model_id

    def get_model_dir(self, model_id: str) -> Path:
        """Return the directory for a specific model.

        Args:
            model_id: The ID of the model.

        Returns:
            The path to the directory for the specified model.
        """
        return self._model_path(model_id)

    def get_variant_dir(self, model_id: str, variant: str) -> Path:
        """Return the directory for a specific model variant.

        Args:
            model_id: The ID of the model.
            variant: The variant of the model.

        Returns:
            The path to the directory for the specified model variant.
        """
        return self._model_path(model_id, variant)

    @overload
    def is_cached(self, model_id: str) -> bool: ...

    @overload
    def is_cached(self, model_id: str, variant: str) -> bool: ...

    def is_cached(self, model_id: str, variant: str | None = None) -> bool:
        """Check if a specific model variant or any variant of a model is cached.

        Args:
            model_id: The ID of the model.
            variant: The variant of the model (optional).

        Returns:
            True if the specified model variant or any variant of the model is cached,
            False otherwise.
        """
        if variant is not None:
            return self._model_path(model_id, variant).exists()
        return self._model_path(model_id).exists()

    def list_cached_models(self) -> list[str]:
        """List all cached model IDs.

        Returns:
            A list of cached model IDs.
        """
        if not self._dir.exists():
            return []
        return [p.name for p in self._dir.iterdir() if p.is_dir()]

    def list_cached_variants(self, model_id: str) -> list[str]:
        """List all cached variants for a specific model.

        Args:
            model_id: The ID of the model.

        Returns:
            A list of cached variants for the specified model.
        """
        model_path = self._model_path(model_id)
        if not model_path.exists():
            return []
        return [p.name for p in model_path.iterdir() if p.is_dir()]

    def list_cache_contents(self) -> dict[str, list[str]]:
        """List all cached models and their variants.

        Returns:
            A dictionary mapping model IDs to lists of their cached variants.
        """
        if not self._dir.exists():
            return {}

        contents: dict[str, list[str]] = {}
        for model_id in self.list_cached_models():
            contents[model_id] = self.list_cached_variants(model_id)

        return contents

    def remove_variant(self, model_id: str, variant: str) -> int:
        """Remove a specific model variant from the cache.

        Args:
            model_id: The ID of the model.
            variant: The variant of the model to remove.

        Returns:
            The total size of the removed files in bytes.
        """
        variant_dir = self._model_path(model_id, variant)
        if not variant_dir.exists():
            msg = f"Model variant directory {variant_dir} does not exist."
            raise FileNotFoundError(msg)

        return clear_directory(variant_dir)

    def remove_model(self, model_id: str) -> int:
        """Remove all variants of a model from the cache.

        Args:
            model_id: The ID of the model to remove.

        Returns:
            The total size of the removed files in bytes.
        """
        model_dir = self._model_path(model_id)
        if not model_dir.exists():
            msg = f"Model directory {model_dir} does not exist."
            raise FileNotFoundError(msg)

        total_size = 0
        for variant_dir in model_dir.iterdir():
            if variant_dir.is_dir():
                total_size += self.remove_variant(model_id, variant_dir.name)
            else:
                log.warning(
                    "Found unexpected file %s in model directory %s during removal.",
                    variant_dir,
                    model_dir,
                )
                total_size += variant_dir.stat().st_size
                variant_dir.unlink()

        model_dir.rmdir()  # Remove the now-empty model directory
        return total_size

    def clear_cache(self) -> int:
        """Clear the entire cache by removing all cached models and variants.

        Returns:
            The total size of the removed files in bytes.
        """
        total_size = 0

        # Clean models
        for model_id in self.list_cached_models():
            total_size += self.remove_model(model_id)

        self._dir.rmdir()  # Remove the now-empty model directory

        return total_size
