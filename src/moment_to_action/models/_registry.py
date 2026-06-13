"""Model registry for moment-to-action."""

# ruff: noqa: ERA001

from ._formats import ModelFormat
from ._model_info import ModelID, ModelInfo
from ._sources import HuggingFaceSource, UltralyticsSource
from .image.classification.mobilenet_v2._model import MobileNetV2Model
from .image.detection.detectron2._model import Detectron2Model
from .image.detection.rf_detr._model import RFDETRModel
from .image.detection.rtmdet._model import RTMDetModel
from .image.detection.yolo._model import YOLOModel

DEFAULT_KEY = "default"

MODEL_REGISTRY: dict[ModelID, ModelInfo] = {
    ModelID.MOBILENET_V2: ModelInfo(
        id=ModelID.MOBILENET_V2,
        model_class=MobileNetV2Model,
        variants={
            DEFAULT_KEY: HuggingFaceSource(
                format=ModelFormat.ONNX,
                hf_repo_id="llamas-lab/m2a-models",
                hf_subdir="mobilenet_v2_onnx",
                files=["model.onnx"],
                revision="515edc0b7e29c1a3c58d2587690e88d5744da530",
            ),
            "qcs6490": HuggingFaceSource(
                format=ModelFormat.DLC,
                hf_repo_id="llamas-lab/m2a-models",
                hf_subdir="mobilenet_v2_qcs",
                files=[
                    "model.dlc",
                    "reference_outputs/inputs.npy",
                    "reference_outputs/outputs_0.npy",
                ],
                revision="f76241f7de7a4942618c95a2429a6c8df8c594ef",
            ),
        },
    ),
    ModelID.YOLO_V8: ModelInfo(
        id=ModelID.YOLO_V8,
        model_class=YOLOModel,
        variants={
            DEFAULT_KEY: UltralyticsSource(
                format=ModelFormat.ONNX,
                name="yolov8n",
            ),
            "qcs6490": HuggingFaceSource(
                format=ModelFormat.DLC,
                hf_repo_id="llamas-lab/m2a-models",
                hf_subdir="yolo_qcs",
                files=[
                    "model.dlc",
                    "model.npu.bin",
                    "labels.txt",
                    "metadata.json",
                    "reference_outputs/inputs.npy",
                    "reference_outputs/outputs_0.npy",
                    "reference_outputs/outputs_1.npy",
                    "reference_outputs/outputs_2.npy",
                ],
                revision="299e2abf9723627d5f9779eb663284598f3bad5c",
            ),
        },
    ),
    ModelID.RF_DETR: ModelInfo(
        id=ModelID.RF_DETR,
        model_class=RFDETRModel,
        variants={
            DEFAULT_KEY: UltralyticsSource(
                format=ModelFormat.ONNX,
                name="rf_detr",
            ),
            # DLC only — float-only model; v68 NPU context binary not feasible.
            "qcs6490": HuggingFaceSource(
                format=ModelFormat.DLC,
                hf_repo_id="llamas-lab/m2a-models",
                hf_subdir="rf_detr_qcs",
                files=[
                    "model.dlc",
                    "metadata.json",
                    "reference_outputs/inputs.npy",
                    "reference_outputs/outputs_0.npy",
                    "reference_outputs/outputs_1.npy",
                    "reference_outputs/outputs_2.npy",
                ],
                revision="062c95760b60e42a64f9b5b65e2921aa629f7ad5",
            ),
        },
    ),
    ModelID.RTM_DET: ModelInfo(
        id=ModelID.RTM_DET,
        model_class=RTMDetModel,
        variants={
            DEFAULT_KEY: UltralyticsSource(
                format=ModelFormat.ONNX,
                name="rtmdet",
            ),
            # DLC only — float decode head; v68 NPU context binary not feasible.
            "qcs6490": HuggingFaceSource(
                format=ModelFormat.DLC,
                hf_repo_id="llamas-lab/m2a-models",
                hf_subdir="rtmdet_qcs",
                files=[
                    "model.dlc",
                    "labels.txt",
                    "metadata.json",
                    "reference_outputs/inputs.npy",
                    "reference_outputs/outputs_0.npy",
                    "reference_outputs/outputs_1.npy",
                    "reference_outputs/outputs_2.npy",
                ],
                revision="062c95760b60e42a64f9b5b65e2921aa629f7ad5",
            ),
        },
    ),
    ModelID.DETECTRON2: ModelInfo(
        id=ModelID.DETECTRON2,
        model_class=Detectron2Model,
        variants={
            DEFAULT_KEY: UltralyticsSource(
                format=ModelFormat.ONNX,
                name="detectron2",
            ),
            "qcs6490_w8a16": HuggingFaceSource(
                format=ModelFormat.DLC,
                hf_repo_id="llamas-lab/m2a-models",
                hf_subdir="detectron2_qcs_w8a16",
                files=[
                    "model.proposal_generator.dlc",
                    "model.proposal_generator.npu.bin",
                    "model.roi_head.dlc",
                    "model.roi_head.npu.bin",
                    "metadata.json",
                    "reference_outputs/inputs.npy",
                    "reference_outputs/outputs_0.npy",
                    "reference_outputs/outputs_1.npy",
                    "reference_outputs/outputs_2.npy",
                ],
                revision="062c95760b60e42a64f9b5b65e2921aa629f7ad5",
            ),
            "qcs6490_w8a8": HuggingFaceSource(
                format=ModelFormat.DLC,
                hf_repo_id="llamas-lab/m2a-models",
                hf_subdir="detectron2_qcs_w8a8",
                files=[
                    "model.proposal_generator.dlc",
                    "model.proposal_generator.npu.bin",
                    "model.roi_head.dlc",
                    "model.roi_head.npu.bin",
                    "metadata.json",
                    "reference_outputs/inputs.npy",
                    "reference_outputs/outputs_0.npy",
                    "reference_outputs/outputs_1.npy",
                    "reference_outputs/outputs_2.npy",
                ],
                revision="062c95760b60e42a64f9b5b65e2921aa629f7ad5",
            ),
        },
        npu_only_variants=frozenset({"qcs6490_w8a16", "qcs6490_w8a8"}),
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
