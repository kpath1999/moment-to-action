"""Model registry — centralized configuration of available models."""

from __future__ import annotations

from ._types import DownloadSource, ModelID, ModelInfo, TransformersSource, VendoredSource

__all__ = ["MODEL_REGISTRY"]


MODEL_REGISTRY: dict[ModelID, ModelInfo] = {
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
    ModelID.QWEN3_4B: ModelInfo(
        id=ModelID.QWEN3_4B,
        filename="__UNUSED__",
        source=TransformersSource(hf_repo_id="Qwen/Qwen3-4B-Instruct-2507"),
    ),
}
