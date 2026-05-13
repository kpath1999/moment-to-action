from enum import Enum
from pathlib import Path

import attrs

from ._sources import ModelSource


class ModelID(Enum):
    """Unique identifier for each model in the registry."""

    YOLO_V8 = "yolo_v8"
    MOBILECLIP_S2 = "mobileclip_s2"
    SMOLVLM2_2_2B = "smolvlm2_2_2b"


@attrs.frozen
class ModelInfo:
    """Static metadata describing a model in the registry."""

    id: ModelID
    """Unique model identifier."""

    variants: dict[str, ModelSource]
    """Dictionary mapping variant names to their respective sources."""


@attrs.frozen
class VariantStatus:
    """Runtime status of a specific model variant."""

    model_id: ModelID
    """Unique identifier of the model this variant belongs to."""

    variant: str
    """Variant name."""

    available: bool
    """Whether this variant is currently available (i.e. downloaded and loadable)."""

    path: Path | None
    """Location of model files; None if not available."""

    size_bytes: int | None
    """Size of the model downloads in bytes if available; None otherwise."""


@attrs.frozen
class ModelStatus:
    """Runtime status of a model."""

    info: ModelInfo
    """Static metadata for this model."""

    variants: list[VariantStatus]
    """List of variant statuses."""

    path: Path | None
    """Location of model files; None if not available."""

    @property
    def size_bytes(self) -> int:
        """Total size of all available variants in bytes, or None if not available."""
        return sum(variant.size_bytes or 0 for variant in self.variants)

    @property
    def available(self) -> bool:
        """Whether the model is available (i.e. at least one variant is available)."""
        return any(variant.available for variant in self.variants)

    @property
    def available_variants(self) -> list[VariantStatus]:
        """List of available variants."""
        return [variant for variant in self.variants if variant.available]
