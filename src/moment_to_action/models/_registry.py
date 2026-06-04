"""Model registry for moment-to-action."""

# ruff: noqa: ERA001

from pathlib import Path

from ._formats import ModelFormat
from ._model_info import ModelID, ModelInfo
from ._sources import HuggingFaceSource, VendoredSource
from .image.detection.yolo._model import YOLOModel

DEFAULT_KEY = "default"

MODEL_REGISTRY: dict[ModelID, ModelInfo] = {
    ModelID.YOLO_V8: ModelInfo(
        id=ModelID.YOLO_V8,
        model_class=YOLOModel,
        variants={
            DEFAULT_KEY: VendoredSource(
                format=ModelFormat.ONNX,
                path=Path("yolo/model.onnx"),
            ),
            "qcs6490": HuggingFaceSource(
                format=ModelFormat.DLC,
                hf_repo_id="llamas-lab/m2a-models",
                hf_subdir="yolo_qcs",
                files=[
                    "model.dlc",
                    "reference_outputs/inputs.npy",
                    "reference_outputs/outputs_0.npy",
                    "reference_outputs/outputs_1.npy",
                    "reference_outputs/outputs_2.npy",
                ],
                revision="3a63631830dec0cbedab34444e7066c524996467",
            ),
        },
    ),
}


# OLD REGISTRY (for reference, to be removed once the new registry is fully implemented):

# MODEL_REGISTRY: dict[ModelID, ModelInfo] = {
#     ModelID.YOLO_V8: ModelInfo(
#         id=ModelID.YOLO_V8,
#         filename="model.onnx",
#         source=VendoredSource(subdir="yolo"),
#     ),
#     ModelID.MOBILECLIP_S2: ModelInfo(
#         id=ModelID.MOBILECLIP_S2,
#         filename="mobileclip_s2_datacompdr_last.tflite",
#         source=DownloadSource(
#             hf_repo_id="anton96vice/mobileclip2_tflite",
#             hf_filename="mobileclip_s2_datacompdr_last.tflite",
#         ),
#     ),
#     ModelID.SMOLVLM2_2_2B: ModelInfo(
#         id=ModelID.SMOLVLM2_2_2B,
#         filename="__UNUSED__",
#         source=TransformersSource(hf_repo_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
#     ),
# }
