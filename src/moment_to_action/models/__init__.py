"""Model management and downloading."""

from ._base import BaseModel
from ._formats import ModelFormat
from ._manager import ModelManager
from ._model_info import ModelID, ModelInfo, Variant, VariantStatus
from ._registry import DEFAULT_KEY as DEFAULT_VARIANT_KEY
from ._registry import MODEL_REGISTRY
from ._sources import (
    DownloadSource,
    HuggingFaceSource,
    ModelSource,
    UltralyticsSource,
    VendoredSource,
    resolve_download_source,
    resolve_hugging_face_source,
    resolve_model_source,
    resolve_ultralytics_source,
    resolve_vendored_source,
)
from .image._base import ImageModel
from .image.classification._base import ImageClassificationModel
from .image.classification._types import Classification
from .image.classification.mobilenet_v2._model import MobileNetV2Model
from .image.detection._base import ImageDetectionModel
from .image.detection._types import BoundingBox, Detection
from .image.detection.rf_detr._model import RFDETRModel
from .image.detection.rtmdet._model import RTMDetModel
from .image.detection.yolo._model import YOLOModel
from .llm._base import LlamaGGUFModel
from .llm.phi35._model import Phi35Model
from .llm.qwen2._model import Qwen2Model
from .llm.qwen3._model import Qwen3Model
from .vlm._base import LlamaVLModel
from .vlm.ministral._model import MinistralVLModel
from .vlm.qwen3_vl._model import Qwen3VLModel
from .vlm.qwen25_vl._model import Qwen25VLModel

__all__ = [
    "DEFAULT_VARIANT_KEY",
    "MODEL_REGISTRY",
    "BaseModel",
    "BoundingBox",
    "Classification",
    "Detection",
    "DownloadSource",
    "HuggingFaceSource",
    "ImageClassificationModel",
    "ImageDetectionModel",
    "ImageModel",
    "LlamaGGUFModel",
    "LlamaVLModel",
    "MinistralVLModel",
    "MobileNetV2Model",
    "ModelFormat",
    "ModelID",
    "ModelInfo",
    "ModelManager",
    "ModelSource",
    "Phi35Model",
    "Qwen2Model",
    "Qwen3Model",
    "Qwen3VLModel",
    "Qwen25VLModel",
    "RFDETRModel",
    "RTMDetModel",
    "UltralyticsSource",
    "Variant",
    "VariantStatus",
    "VendoredSource",
    "YOLOModel",
    "resolve_download_source",
    "resolve_hugging_face_source",
    "resolve_model_source",
    "resolve_ultralytics_source",
    "resolve_vendored_source",
]
