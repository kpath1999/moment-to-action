import logging
from pathlib import Path
from typing import Protocol

import attrs
import rich
import rich.table
from humanfriendly import format_size

from moment_to_action.utils.files import clear_directory, disk_size

from ._models import ModelCacheContents, ModelCacheManager

log = logging.getLogger(__name__)


class _SubcacheInfo(Protocol):
    """Subcache information."""

    @property
    def total_size_bytes(self) -> int: ...

    @property
    def item_count(self) -> int: ...


@attrs.frozen
class CacheInfo:
    """Information about the cache, including total size and details about sub-caches."""

    total_size_bytes: int
    """The total size of the cache in bytes."""

    root_contents: list[Path]
    """List of files/directories directly under the cache directory."""

    models_info: ModelCacheContents
    """Information about the cached models."""

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the cache info."""
        return {
            "total_size_bytes": self.total_size_bytes,
            "root_contents": [str(p) for p in self.root_contents],
            "models_info": self.models_info.to_json(),
        }

    def to_rich_table(self) -> rich.table.Table:
        """Return a Rich table summarizing the cache contents."""
        # Title
        table = rich.table.Table(
            "Subcache",
            "Size",
            "Items",
            title=f"Cache ({format_size(self.total_size_bytes)})",
        )

        # Subcaches
        def add_subcache(name: str, subcache: _SubcacheInfo) -> None:
            table.add_row(name, format_size(subcache.total_size_bytes), str(subcache.item_count))

        add_subcache("models", self.models_info)

        # Root
        if self.root_contents:
            table.add_section()
            table.add_row(
                "[dim]other[/dim]",
                "",
                ", ".join(str(p) for p in self.root_contents),
            )
        return table


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

    def inspect_cache(self) -> CacheInfo:
        """Inspect the current state of the cache.

        Returns:
            Information about the cache.
        """
        known_sub_caches = {"models"}

        # Get info about sub caches
        models_info = self._model_manager.list_cache_contents()

        # Get info about root files
        root_contents: list[Path] = [
            i for i in self._cache_dir.iterdir() if i.name not in known_sub_caches
        ]

        # Compute total size
        total_size = sum(disk_size(p) for p in root_contents) + models_info.total_size_bytes

        return CacheInfo(
            total_size_bytes=total_size, root_contents=root_contents, models_info=models_info
        )

    def clear_cache(self) -> CacheInfo:
        """Clear the entire cache directory.

        Returns:
            Information about what was cleared.
        """
        size = 0

        # Clear models
        models_info = self._model_manager.clear_cache()
        size += models_info.total_size_bytes

        # Ensure the cache directory is empty
        other_files: list[Path] = []

        for item in self._cache_dir.iterdir():
            log.warning("Found unexpected item %s in cache directory during clearing.", item)
            other_files.append(item)

            if item.is_file():
                size += item.stat().st_size
                item.unlink()
            elif item.is_dir():
                size += clear_directory(item)

        return CacheInfo(total_size_bytes=size, root_contents=other_files, models_info=models_info)
