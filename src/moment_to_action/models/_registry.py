"""Model registry — centralized configuration of available models."""

from __future__ import annotations

from ._types import DownloadSource, ModelID, ModelInfo, TransformersSource

__all__ = ["MODEL_REGISTRY"]


MODEL_REGISTRY: dict[ModelID, ModelInfo] = {
    # Target detection model IDs for the COCO benchmark track.
    ModelID.YOLO_V12_N: ModelInfo(
        id=ModelID.YOLO_V12_N,
        filename="yolo12n.onnx",
        source=DownloadSource(
            hf_repo_id="webnn/yolo12n",
            hf_filename="onnx/yolo12n.onnx",
        ),
    ),
    ModelID.RF_DETR_N: ModelInfo(
        id=ModelID.RF_DETR_N,
        filename="rf_detr_n.onnx",
        source=DownloadSource(
            hf_repo_id="onnx-community/rfdetr_nano-ONNX",
            hf_filename="onnx/model.onnx",
        ),
    ),
    ModelID.SSD_MOBILENETV2: ModelInfo(
        id=ModelID.SSD_MOBILENETV2,
        filename="ssd-mobilenet-v2.onnx",
        source=DownloadSource(
            hf_repo_id="Kalray/ssd-mobilenet-v2",
            hf_filename="ssd-mobilenet-v2.onnx",
        ),
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
