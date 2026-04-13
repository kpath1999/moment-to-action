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
        filename="yolov8_det.tflite",
        source=VendoredSource(subdir="yolo_tflite"),
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

    ModelID.QWEN_2_5: ModelInfo(
        id=ModelID.QWEN_2_5,
        #filename="qwen2.5-1.5b-instruct-q5_k_m.gguf",
        filename="qwen2-1_5b-instruct-q4_0-pure.gguf",
        #filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        source=VendoredSource(subdir="slm_models")
    ),
}
