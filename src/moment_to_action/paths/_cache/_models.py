from __future__ import annotations

import logging
from typing import TYPE_CHECKING, overload

import attrs
import rich
import rich.table
from humanfriendly import format_size

from moment_to_action.utils.files import clear_directory, disk_size

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


@attrs.frozen
class CachedModelInfo:
    """Information about a cached model in the cache."""

    model_id: str
    """The ID of the cached model."""

    size_bytes: int
    """The total size of the cached model in bytes."""

    variants: list[str]
    """List of variants that are cached for this model."""

    other: list[Path]
    """Other unexpected files found while inspecting the model."""

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the cached model info."""
        return {
            **attrs.asdict(self),
            "other": [str(p) for p in self.other],  # paths are not json serializable
        }

    def to_rich_table_row(self) -> list[object]:
        """Return a list of values representing this model info for display in a rich table.

        Columns: [model_id, size, variants, dirty Y/N]
        """
        dirty_count = len(self.other)
        return [
            self.model_id,
            format_size(self.size_bytes),
            ", ".join(self.variants),
            f"[red]{dirty_count}[/red]" if dirty_count else "[green]0[/green]",
        ]


@attrs.frozen
class ModelCacheContents:
    """List information about the model cache contents."""

    total_size_bytes: int
    """Total size of the models cache."""

    models: dict[str, CachedModelInfo]
    """Map from model ID to information about the model."""

    other: list[Path]
    """Other (unexpected) files found."""

    @property
    def item_count(self) -> int:
        """Return the number fo items in this cache."""
        return len(self.models)

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the cache contents."""
        return {
            "total_size_bytes": self.total_size_bytes,
            "models": {k: v.to_json() for k, v in self.models.items()},
            "other": [str(p) for p in self.other],
        }

    def models_to_rich_table(self) -> rich.table.Table:
        """Return a Rich table populated with all cached model info."""
        table = rich.table.Table(
            "Model ID",
            "Size",
            "Variants",
            "Dirty Files",
            title=f"Model Cache ({format_size(self.total_size_bytes)})",
        )

        # Add models
        for model in self.models.values():
            table.add_row(*[str(c) for c in model.to_rich_table_row()])

        # Footer row with other fiels
        if self.other:
            table.add_section()
            table.add_row(
                "[dim]other[/dim]",
                "",
                ", ".join(str(p) for p in self.other),
                "",
            )
        return table


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

    def _cached_model_paths(self) -> list[Path]:
        """Get the paths to all cached models."""
        return [p for p in self._dir.iterdir() if p.is_dir()]

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

    def list_cached_models(self) -> dict[str, CachedModelInfo]:
        """List all cached models and their variants.

        Returns:
            A dictionary mapping model IDs to information about them.
        """
        if not self._dir.exists():
            return {}

        def build_info(p: Path) -> CachedModelInfo:
            """Build info for a model at this path."""
            variants = self.list_cached_variants(p.name)
            variants_set = frozenset(variants)

            return CachedModelInfo(
                model_id=p.name,
                size_bytes=disk_size(p),
                variants=variants,
                other=[f for f in p.iterdir() if f not in variants_set],
            )

        return {p.name: build_info(p) for p in self._cached_model_paths()}

    def list_cache_contents(self) -> ModelCacheContents:
        """List conents of the models cache.

        Included models and other (unexpected) files found.

        Returns:
            Information about the contents of the model cache.
        """
        # Get models
        models = self.list_cached_models()

        # Look at other files
        files = [p for p in self._dir.iterdir() if p.name not in models]

        # Get total size
        total_size = sum(disk_size(p) for p in files) + sum(m.size_bytes for m in models.values())

        # Done!
        return ModelCacheContents(total_size_bytes=total_size, models=models, other=files)

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

    def remove_model(self, model_id: str) -> CachedModelInfo:
        """Remove all variants of a model from the cache.

        Args:
            model_id: The ID of the model to remove.

        Returns:
            Information about the removed model, including total size and variants.
        """
        model_dir = self._model_path(model_id)
        if not model_dir.exists():
            msg = f"Model directory {model_dir} does not exist."
            raise FileNotFoundError(msg)

        total_size = 0
        variants: list[str] = []
        other_files: list[Path] = []

        for variant_dir in model_dir.iterdir():
            if variant_dir.is_dir():
                total_size += self.remove_variant(model_id, variant_dir.name)
                variants.append(variant_dir.name)
            else:
                log.warning(
                    "Found unexpected file %s in model directory %s during removal.",
                    variant_dir,
                    model_dir,
                )
                total_size += variant_dir.stat().st_size
                variant_dir.unlink()

                other_files.append(variant_dir)

        model_dir.rmdir()  # Remove the now-empty model directory
        return CachedModelInfo(
            model_id=model_id, size_bytes=total_size, variants=variants, other=other_files
        )

    def clear_cache(self) -> ModelCacheContents:
        """Clear the entire cache by removing all cached models and variants.

        Also removes any other files that may have unexpectedly shown up.

        Returns:
            Information about the model cache contents.
        """
        total_size = 0

        # Clean models
        models_info: dict[str, CachedModelInfo] = {}

        for model_path in self._cached_model_paths():
            model_id = model_path.name

            info = self.remove_model(model_id=model_id)
            total_size += info.size_bytes

            models_info[model_id] = info

        # Ensure there's nothing left in the model cache directory
        other_files: list[Path] = []

        for item in self._dir.iterdir():
            log.warning("Found unexpected item %s in model cache directory during clearing.", item)
            other_files.append(item)

            if item.is_file():
                total_size += item.stat().st_size
                item.unlink()
            elif item.is_dir():
                total_size += clear_directory(item)

        self._dir.rmdir()  # Remove the now-empty model directory

        return ModelCacheContents(
            total_size_bytes=total_size, models=models_info, other=other_files
        )
