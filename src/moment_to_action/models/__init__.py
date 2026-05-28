"""Model management and downloading."""

from ._base import BaseModel
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
from .image._base import ImageModel
from .image.detection._base import ImageDetectionModel
from .image.detection._types import BoundingBox, Detection
from .image.detection.yolo._model import YOLOModel

__all__ = [
    "DEFAULT_VARIANT_KEY",
    "MODEL_REGISTRY",
    "BaseModel",
    "BoundingBox",
    "Detection",
    "DownloadSource",
    "HuggingFaceSource",
    "ImageDetectionModel",
    "ImageModel",
    "ModelFormat",
    "ModelID",
    "ModelInfo",
    "ModelManager",
    "ModelSource",
    "VariantStatus",
    "VendoredSource",
    "YOLOModel",
    "resolve_download_source",
    "resolve_hugging_face_source",
    "resolve_model_source",
    "resolve_vendored_source",
]
