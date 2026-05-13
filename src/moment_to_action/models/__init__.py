"""Model management and downloading."""

from ._formats import ModelFormat
from ._manager import ModelManager
from ._model_info import ModelID, ModelInfo, VariantStatus
from ._registry import DEFAULT_KEY as DEFAULT_VARIANT_KEY
from ._registry import MODEL_REGISTRY
from ._sources import (
    DownloadSource,
    HuggingFaceSource,
    ModelSource,
    VendoredSource,
    resolve_download_source,
    resolve_hugging_face_source,
    resolve_model_source,
    resolve_vendored_source,
)

__all__ = [
    "DEFAULT_VARIANT_KEY",
    "MODEL_REGISTRY",
    "DownloadSource",
    "HuggingFaceSource",
    "ModelFormat",
    "ModelID",
    "ModelInfo",
    "ModelInfo",
    "ModelManager",
    "ModelSource",
    "VariantStatus",
    "VendoredSource",
    "resolve_download_source",
    "resolve_hugging_face_source",
    "resolve_model_source",
    "resolve_vendored_source",
]
