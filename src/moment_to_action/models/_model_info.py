from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from pathlib import Path

    from moment_to_action.hardware._types import ComputeUnit

    from ._base import BaseModel
    from ._sources import ModelSource


class ModelID(Enum):
    """Unique identifier for each model in the registry."""

    YOLO_V8 = "yolo_v8"
    MOBILECLIP_S2 = "mobileclip_s2"
    SMOLVLM2_2_2B = "smolvlm2_2_2b"
    MOBILENET_V2 = "mobilenet_v2"
    RF_DETR = "rf_detr"
    RTM_DET = "rtm_det"
    DETECTRON2 = "detectron2"
    QWEN2_1_5B_INSTRUCT = "qwen2_1_5b_instruct"
    QWEN2_7B_INSTRUCT = "qwen2_7b_instruct"
    QWEN3_4B = "qwen3_4b"
    PHI35_MINI_INSTRUCT = "phi35_mini_instruct"
    QWEN25_VL_3B_INSTRUCT = "qwen25_vl_3b_instruct"
    QWEN25_VL_7B_INSTRUCT = "qwen25_vl_7b_instruct"
    QWEN3_VL_2B_INSTRUCT = "qwen3_vl_2b_instruct"
    QWEN3_VL_4B_INSTRUCT = "qwen3_vl_4b_instruct"
    MINISTRAL_3B_INSTRUCT = "ministral_3b_instruct"


@attrs.frozen
class Variant:
    """A specific model variant with its source, supported backends, and input layout.

    The ``backends`` dict is the single authoritative declaration of which compute
    units this variant supports and which artifact files each unit loads.  Keys
    present in ``backends`` are the only units this variant can run on; anything
    absent will be caught at load time.

    Args:
        source: Download/vendored source for this variant's files.
        backends: Mapping of compute unit to ``{component_name: filename}`` dicts.
            Component key is ``"model"`` for single-graph models;
            ``"proposal_generator"``/``"roi_head"`` for two-component Detectron2.
            Filenames are relative to the variant directory.
        input_layout: Input tensor layout — ``"NCHW"``, ``"NHWC"``, or ``None``
            for model types that do not require a spatial layout (e.g. LLMs).
            Image model constructors default to ``"NCHW"`` when ``None`` is passed.
    """

    source: ModelSource
    """Download/vendored source for this variant's files."""

    backends: dict[ComputeUnit, dict[str, str]]
    """Compute unit → component filename mapping; keys = supported units."""

    input_layout: str | None = None
    """Input tensor layout: ``"NCHW"``, ``"NHWC"``, or ``None`` (not applicable)."""


@attrs.frozen
class ModelInfo:
    """Static metadata describing a model in the registry."""

    id: ModelID
    """Unique model identifier."""

    variants: dict[str, Variant]
    """Dictionary mapping variant names to their :class:`Variant` descriptors."""

    model_class: type[BaseModel]
    """Concrete model class to instantiate when
    :meth:`~moment_to_action.models.ModelManager.get_model` is called."""


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
