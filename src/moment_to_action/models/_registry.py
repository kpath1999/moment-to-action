"""Model registry — centralized configuration of available models."""

from __future__ import annotations

from ._types import DownloadSource, ModelID, AssetID, ModelInfo, AssetInfo, TransformersSource, VendoredSource

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
        #filename="gear_guard_net-ppe-detection-w8a8.tflite",
        source=VendoredSource(subdir="yolo_tflite"),
        #source=VendoredSource(subdir="yolo_ppe"),
    ),    
    ModelID.MOBILECLIP_S2: ModelInfo(
        id=ModelID.MOBILECLIP_S2,
        filename="mobileclip_s2_datacompdr_last.tflite",
        source=DownloadSource(
            hf_repo_id="anton96vice/mobileclip2_tflite",
            hf_filename="mobileclip_s2_datacompdr_last.tflite",
        ),
    ),
    ModelID.MOBILECLIP_S2_IMAGE: ModelInfo(
        id=ModelID.MOBILECLIP_S2_IMAGE,
        filename="mobileclip_image_gpu_b8_float32.tflite",
        source=VendoredSource(subdir="vlm_models/mobileclip_mod/image/")
    ),
    ModelID.MOBILECLIP_S2_BATCHED: ModelInfo(
        id=ModelID.MOBILECLIP_S2_BATCHED,
        filename="mobileclip_s2_datacompdr_last_patched_float32.tflite",
        source=VendoredSource(subdir="vlm_models/mobileclip-s2_batched/")
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
    ModelID.YAMNET_TFLITE: ModelInfo(
        id=ModelID.YAMNET_TFLITE,
        filename="yamnet.tflite",
        source=VendoredSource(subdir="yamnet"),
    ),
    ModelID.YAMNET_ONNX: ModelInfo(
        id=ModelID.YAMNET_ONNX,
        filename="model.onnx",
        source=VendoredSource(subdir="yamnet"),
    ),    
}

ASSET_REGISTRY: dict[AssetID, AssetInfo] = {
    AssetID.YAMNET_CLASS_MAP: AssetInfo(
        id=AssetID.YAMNET_CLASS_MAP,
        filename="yamnet_class_map.csv",
        source=VendoredSource(subdir="yamnet"),
    ),
}
