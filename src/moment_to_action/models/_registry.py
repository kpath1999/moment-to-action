"""Model registry — centralized configuration of available models."""

from __future__ import annotations

from ._types import DownloadSource, ModelID, ModelInfo, TransformersSource, VendoredSource

__all__ = ["MODEL_REGISTRY"]


MODEL_REGISTRY: dict[ModelID, ModelInfo] = {
    # Target detection model IDs for the COCO benchmark track.
    # YOLOv12-n is currently wired to the existing vendored YOLO artifacts
    # until dedicated v12 assets are added.
    ModelID.YOLO_V12_N: ModelInfo(
        id=ModelID.YOLO_V12_N,
        filename="model.onnx",
        source=VendoredSource(subdir="yolo"),
    ),
    ModelID.YOLO_V12_N_TFLITE: ModelInfo(
        id=ModelID.YOLO_V12_N_TFLITE,
        filename="model.tflite",
        source=VendoredSource(subdir="yolo"),
    ),
    ModelID.YOLO_V12_N_TFLITE_INT8: ModelInfo(
        id=ModelID.YOLO_V12_N_TFLITE_INT8,
        filename="model_int8.tflite",
        source=VendoredSource(subdir="yolo"),
    ),
    ModelID.YOLO_V12_N_TFLITE_INT8_320: ModelInfo(
        id=ModelID.YOLO_V12_N_TFLITE_INT8_320,
        filename="model_int8_320.tflite",
        source=VendoredSource(subdir="yolo"),
    ),
    ModelID.YOLO_V8: ModelInfo(
        id=ModelID.YOLO_V8,
        filename="model.onnx",
        source=VendoredSource(subdir="yolo"),
    ),
    ModelID.YOLO_V8_TFLITE: ModelInfo(
        id=ModelID.YOLO_V8_TFLITE,
        filename="model.tflite",
        source=VendoredSource(subdir="yolo"),
    ),
    ModelID.YOLO_V8_TFLITE_INT8: ModelInfo(
        id=ModelID.YOLO_V8_TFLITE_INT8,
        filename="model_int8.tflite",
        source=VendoredSource(subdir="yolo"),
    ),
    ModelID.YOLO_V8_TFLITE_INT8_320: ModelInfo(
        id=ModelID.YOLO_V8_TFLITE_INT8_320,
        filename="model_int8_320.tflite",
        source=VendoredSource(subdir="yolo"),
    ),
    ModelID.RF_DETR_N: ModelInfo(
        id=ModelID.RF_DETR_N,
        filename="__UNUSED__",
        source=TransformersSource(hf_repo_id="PekingU/rf-detr-base"),
    ),
    ModelID.SSD_MOBILENETV2: ModelInfo(
        id=ModelID.SSD_MOBILENETV2,
        filename="__UNUSED__",
        source=TransformersSource(hf_repo_id="Narsil/ssd_mobilenet_v2"),
    ),
    # Target retrieval model IDs for the COCO benchmark track.
    ModelID.TINYCLIP_8M: ModelInfo(
        id=ModelID.TINYCLIP_8M,
        filename="__UNUSED__",
        source=TransformersSource(hf_repo_id="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"),
    ),
    ModelID.MOBILECLIP_S2: ModelInfo(
        id=ModelID.MOBILECLIP_S2,
        filename="mobileclip_s2_datacompdr_last.tflite",
        source=DownloadSource(
            hf_repo_id="anton96vice/mobileclip2_tflite",
            hf_filename="mobileclip_s2_datacompdr_last.tflite",
        ),
    ),
    ModelID.SMOLVLM2_2_2B: ModelInfo(
        id=ModelID.SMOLVLM2_2_2B,
        filename="__UNUSED__",
        source=TransformersSource(hf_repo_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
    ),
    ModelID.QWEN2_5_4B: ModelInfo(
        id=ModelID.QWEN2_5_4B,
        filename="__UNUSED__",
        source=TransformersSource(hf_repo_id="Qwen/Qwen2.5-4B-Instruct"),
    ),
    ModelID.WHISPER_TINY: ModelInfo(
        id=ModelID.WHISPER_TINY,
        filename="__UNUSED__",
        source=TransformersSource(hf_repo_id="openai/whisper-tiny"),
    ),
    ModelID.GROUNDING_DINO_BASE: ModelInfo(
        id=ModelID.GROUNDING_DINO_BASE,
        filename="__UNUSED__",
        source=TransformersSource(hf_repo_id="IDEA-Research/grounding-dino-base"),
    ),
    ModelID.SIGLIP_SO400M: ModelInfo(
        id=ModelID.SIGLIP_SO400M,
        filename="__UNUSED__",
        source=TransformersSource(hf_repo_id="google/siglip-so400m-patch14-384"),
    ),
}
