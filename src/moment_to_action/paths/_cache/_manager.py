import logging
from pathlib import Path

from moment_to_action.utils.files import clear_directory

from ._models import ModelCacheManager

log = logging.getLogger(__name__)


class CacheManager:
    """Manager for the application cache on disk."""

    def __init__(self, cache_dir: Path) -> None:
        """Initialize the cache manager with the given cache directory.

        Args:
            cache_dir: The directory where cache files will be stored.
        """
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Create submanagers
        self._model_manager = ModelCacheManager(self._cache_dir / "models")

    @property
    def cache_dir(self) -> Path:
        """Return the cache directory."""
        return self._cache_dir

    @property
    def models(self) -> ModelCacheManager:
        """Return the model cache manager."""
        return self._model_manager

    def clear_cache(self) -> int:
        """Clear the entire cache directory.

        Returns:
            The total size of the cleared cache in bytes.
        """
        size = 0

        # Clear models
        size += self._model_manager.clear_cache()

        # Ensure the cache directory is empty
        for item in self._cache_dir.iterdir():
            log.warning("Found unexpected item %s in cache directory during clearing.", item)
            if item.is_file():
                size += item.stat().st_size
                item.unlink()
            elif item.is_dir():
                size += clear_directory(item)

        return size
