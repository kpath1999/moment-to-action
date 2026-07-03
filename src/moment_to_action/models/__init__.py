"""Model management and downloading."""

from ._base import BaseModel
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
from .llm.gemma3._model import Gemma3Model
from .llm.phi35._model import Phi35Model
from .llm.qwen2._model import Qwen2Model
from .llm.qwen3._model import Qwen3Model
from .vlm._base import LlamaVLModel
from .vlm.internvl3._model import InternVL3Model
from .vlm.ministral._model import MinistralModel
from .vlm.moondream2._model import Moondream2Model
from .vlm.qwen3_vl._model import Qwen3VLModel
from .vlm.qwen25_vl._model import Qwen25VLModel
from .vlm.smolvlm2._model import SmolVLM2Model

__all__ = [
    "DEFAULT_VARIANT_KEY",
    "MODEL_REGISTRY",
    "BaseModel",
    "BoundingBox",
    "Classification",
    "Detection",
    "DownloadSource",
    "Gemma3Model",
    "HuggingFaceSource",
    "ImageClassificationModel",
    "ImageDetectionModel",
    "ImageModel",
    "InternVL3Model",
    "LlamaGGUFModel",
    "LlamaVLModel",
    "MinistralModel",
    "MobileNetV2Model",
    "ModelID",
    "ModelInfo",
    "ModelManager",
    "ModelSource",
    "Moondream2Model",
    "Phi35Model",
    "Qwen2Model",
    "Qwen3Model",
    "Qwen3VLModel",
    "Qwen25VLModel",
    "RFDETRModel",
    "RTMDetModel",
    "SmolVLM2Model",
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
