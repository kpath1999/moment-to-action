from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from moment_to_action.utils.files import disk_size

from ._model_info import ModelID, ModelInfo, ModelStatus, Variant, VariantStatus
from ._registry import DEFAULT_KEY, MODEL_REGISTRY
from ._sources import resolve_model_source

if TYPE_CHECKING:
    from pathlib import Path

    from moment_to_action.hardware._types import ComputeUnit
    from moment_to_action.paths import PathManager
    from moment_to_action.paths._cache._models import CachedModelInfo, ModelCacheContents

    from ._base import BaseModel

log = logging.getLogger(__name__)


class ModelManager:
    """Manager for handling model metadata and availability."""

    def __init__(
        self,
        path_manager: PathManager,
        *,
        registry: dict[ModelID, ModelInfo] = MODEL_REGISTRY,
        show_progress: bool = True,
    ) -> None:
        """Initialize the ModelManager with a given registry.

        Args:
            path_manager: An instance of PathManager to handle file paths.
            registry: A dictionary mapping ModelIDs to their ModelInfo. Defaults to MODEL_REGISTRY.
            show_progress: Whether to show progress bars during downloads.
        """
        self._registry = registry
        self._path_manager = path_manager
        self._show_progress = show_progress

    def _get_model_info(self, model: ModelID) -> ModelInfo:
        """Retrieve the ModelInfo for a given ModelID."""
        if model not in self._registry:
            msg = f"Model {model} not found in registry."
            raise ValueError(msg)
        return self._registry[model]

    @staticmethod
    def _effective_variant(model_info: ModelInfo, variant: str, unit: ComputeUnit | None) -> str:
        """Resolve the variant to actually load for a target compute unit.

        If the requested variant supports ``unit``, it is returned unchanged.
        If not and the ``default`` variant supports ``unit``, a redirect to
        ``default`` is logged and returned.  Otherwise the original variant is
        returned verbatim (``load()`` will raise a clear error at that point).

        Args:
            model_info: Static metadata for the model being resolved.
            variant: The variant the caller requested.
            unit: Target compute unit, or ``None`` to disable redirection
                (the variant is returned verbatim).

        Returns:
            The variant key to load.
        """
        if unit is None:
            return variant
        var_obj = model_info.variants.get(variant)
        if var_obj is None:
            return variant
        if unit in var_obj.backends:
            return variant
        default_var = model_info.variants.get(DEFAULT_KEY)
        if default_var is not None and unit in default_var.backends and variant != DEFAULT_KEY:
            log.info(
                "Variant '%s' of %s is unsupported on %s; using '%s'.",
                variant,
                model_info.id,
                unit,
                DEFAULT_KEY,
            )
            return DEFAULT_KEY
        return variant

    @staticmethod
    def _get_variant(model_info: ModelInfo, variant: str) -> Variant:
        """Get the Variant descriptor for a specific variant key.

        Args:
            model_info: Static metadata for the model being resolved.
            variant: Variant key to look up.

        Returns:
            The :class:`Variant` descriptor for the requested key.

        Raises:
            ValueError: If ``variant`` is not registered for this model.
        """
        if variant not in model_info.variants:
            msg = f"Variant '{variant}' not found for model {model_info.id}."
            raise ValueError(msg)
        return model_info.variants[variant]

    def _get_model_cache_dir(self, model: ModelID, variant: str) -> Path:
        """Get the cache directory for a specific model and variant."""
        return self._path_manager.cache.models.get_variant_dir(model.value, variant)

    def _resolve_model(self, model_id: ModelID, variant: str, *, download: bool) -> Path | None:
        """Resolve the model source to a local file path, downloading if necessary."""
        info = self._get_model_info(model_id)
        variant_obj = self._get_variant(info, variant)

        model_dir = self._get_model_cache_dir(model_id, variant)
        if download:
            # Ensure the cache directory exists before downloading
            model_dir.mkdir(parents=True, exist_ok=True)

        return resolve_model_source(
            variant_obj.source, model_dir, download=download, progress=self._show_progress
        )

    @staticmethod
    def _available(path: Path | None) -> bool:
        """Availibility check for return of resolve_model_source."""
        if path is None:
            return False

        # Path must exist now (invariant)
        if not path.exists():
            msg = f"Expected model file at {path} does not exist."
            raise RuntimeError(msg)

        return True

    def get_path(
        self, model: ModelID, variant: str = DEFAULT_KEY, unit: ComputeUnit | None = None
    ) -> Path:
        """Get the file path for a given model and variant, downloading if necessary.

        Args:
            model: The ModelID of the desired model.
            variant: The variant name of the model to retrieve.
            unit: Target compute unit.  When set, a variant that does not
                support ``unit`` is redirected to ``default`` if ``default``
                does support it (see :meth:`_effective_variant`).

        Returns:
            Path to the model file(s).
        """
        variant = self._effective_variant(self._get_model_info(model), variant, unit)
        # Resolve the model source to get the path
        path = self._resolve_model(model, variant, download=True)

        if not self._available(path):
            msg = f"Download succeeded but model file not found at expected location: {path}"
            raise RuntimeError(msg)

        assert path is not None  # noqa: S101 # For type checker, path should be non-None if available

        # Done!
        log.info("Model %s (variant '%s') is available at: %s", model, variant, path)
        return path

    def is_available(self, model: ModelID, variant: str = DEFAULT_KEY) -> bool:
        """Check if a given model and variant is available locally.

        Args:
            model: The ModelID of the desired model.
            variant: The variant name of the model to check.

        Returns:
            True if the model file(s) exist locally, False otherwise.
        """
        # Resolve the model source to get the expected path
        path = self._resolve_model(model, variant, download=False)
        available = self._available(path)

        log.debug("Model %s (variant '%s') availability: %s at %s", model, variant, available, path)
        return available

    def list_models(self) -> list[ModelStatus]:
        """Return status of all known models without downloading.

        Reports the current availability of each model, along with paths and sizes if downloaded.

        Returns:
            A list of ModelStatus objects representing the status of each model.
        """
        statuses: list[ModelStatus] = []
        for model_id, info in self._registry.items():
            # Find availalbe variants for this model
            variant_statuses: list[VariantStatus] = []
            for variant in info.variants:
                variant_path = self._resolve_model(model_id, variant, download=False)
                available = self._available(variant_path)

                if available:
                    assert variant_path is not None  # noqa: S101  # For type checker, should be non-None if available
                    size_bytes = disk_size(variant_path)
                else:
                    size_bytes = None

                # Build status
                variant_status = VariantStatus(
                    model_id=model_id,
                    variant=variant,
                    available=available,
                    path=variant_path,
                    size_bytes=size_bytes,
                )
                variant_statuses.append(variant_status)

            # Get overall model path if any variant is available
            if any(v.available for v in variant_statuses):
                model_path = self._path_manager.cache.models.get_model_dir(model_id.value)
            else:
                model_path = None

            # Build model status
            model_status = ModelStatus(
                info=info,
                variants=variant_statuses,
                path=model_path,
            )
            statuses.append(model_status)

        return statuses

    def get_model(
        self,
        model_id: ModelID,
        *,
        variant: str = DEFAULT_KEY,
        unit: ComputeUnit | None = None,
        **model_kwargs: object,
    ) -> BaseModel:
        """Construct an unloaded model instance for the given model and variant.

        The model path is resolved (downloading if necessary) before the
        instance is created.  The caller is responsible for calling
        ``model.load(backend)`` before running inference.

        Args:
            model_id: Identifier of the desired model.
            variant: Variant name to load.  Defaults to :data:`DEFAULT_KEY`.
            unit: Target compute unit the model will run on.  When set, a
                variant that does not support ``unit`` is redirected to
                ``default`` if ``default`` does support it, so the returned
                model uses the correct format for the actual variant loaded.
                ``None`` disables redirection.
            **model_kwargs: Additional keyword arguments forwarded verbatim to
                the model constructor.  Use for model-specific parameters such
                as ``confidence_threshold`` on detection models.

        Returns:
            An unloaded :class:`~moment_to_action.models._base.BaseModel` subclass instance.
        """
        info = self._get_model_info(model_id)
        variant = self._effective_variant(info, variant, unit)
        variant_obj = self._get_variant(info, variant)
        path = self.get_path(model_id, variant)
        return info.model_class(
            variant,
            path,
            variant_obj.model_type,
            variant_obj.data_type,
            backends=variant_obj.backends,
            input_layout=variant_obj.input_layout,
            **model_kwargs,
        )

    def remove_variant(self, model_id: ModelID, variant: str) -> int:
        """Remove a specific cached model variant.

        Args:
            model_id: The ModelID of the model to remove.
            variant: Variant key to remove.

        Returns:
            Number of bytes freed.

        Raises:
            FileNotFoundError: If the variant directory does not exist.
        """
        return self._path_manager.cache.models.remove_variant(model_id.value, variant)

    def remove_model(self, model_id: ModelID) -> CachedModelInfo:
        """Remove all cached variants of a model.

        Args:
            model_id: The ModelID of the model to remove.

        Returns:
            :class:`~moment_to_action.paths._cache._models.CachedModelInfo`
            describing the removed model and bytes freed.

        Raises:
            FileNotFoundError: If the model directory does not exist.
        """
        return self._path_manager.cache.models.remove_model(model_id.value)

    def clear_cache(self) -> ModelCacheContents:
        """Clear all downloaded model files from the cache.

        Returns:
            Information about the cleared model cache contents.
        """
        return self._path_manager.cache.models.clear_cache()
