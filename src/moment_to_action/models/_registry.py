"""Model registry for moment-to-action."""

from moment_to_action.hardware._types import ComputeUnit

from ._formats import ModelFormat
from ._model_info import ModelID, ModelInfo, Variant
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
            DEFAULT_KEY: Variant(
                source=HuggingFaceSource(
                    format=ModelFormat.ONNX,
                    hf_repo_id="llamas-lab/m2a-models",
                    hf_subdir="mobilenet_v2_onnx",
                    files=["model.onnx"],
                    revision="515edc0b7e29c1a3c58d2587690e88d5744da530",
                ),
                backends={
                    ComputeUnit.CPU: {"model": "model.onnx"},
                    ComputeUnit.GPU: {"model": "model.onnx"},
                },
            ),
            "qcs6490": Variant(
                source=HuggingFaceSource(
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
                backends={
                    ComputeUnit.CPU: {"model": "model.dlc"},
                    ComputeUnit.GPU: {"model": "model.dlc"},
                    ComputeUnit.NPU: {"model": "model.dlc"},
                },
            ),
        },
    ),
    ModelID.YOLO_V8: ModelInfo(
        id=ModelID.YOLO_V8,
        model_class=YOLOModel,
        variants={
            DEFAULT_KEY: Variant(
                source=UltralyticsSource(
                    format=ModelFormat.ONNX,
                    name="yolov8n",
                ),
                backends={
                    ComputeUnit.CPU: {"model": "model.onnx"},
                    ComputeUnit.GPU: {"model": "model.onnx"},
                },
            ),
            "qcs6490": Variant(
                source=HuggingFaceSource(
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
                backends={
                    ComputeUnit.CPU: {"model": "model.dlc"},
                    ComputeUnit.GPU: {"model": "model.dlc"},
                    ComputeUnit.NPU: {"model": "model.npu.bin"},
                },
                input_layout="NHWC",
            ),
        },
    ),
    ModelID.RF_DETR: ModelInfo(
        id=ModelID.RF_DETR,
        model_class=RFDETRModel,
        variants={
            DEFAULT_KEY: Variant(
                source=UltralyticsSource(
                    format=ModelFormat.ONNX,
                    name="rf_detr",
                ),
                backends={
                    ComputeUnit.CPU: {"model": "model.onnx"},
                    ComputeUnit.GPU: {"model": "model.onnx"},
                },
            ),
            # DLC only — float-only model; v68 NPU context binary not feasible.
            "qcs6490": Variant(
                source=HuggingFaceSource(
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
                backends={
                    ComputeUnit.CPU: {"model": "model.dlc"},
                    ComputeUnit.GPU: {"model": "model.dlc"},
                },
                input_layout="NHWC",
            ),
        },
    ),
    ModelID.RTM_DET: ModelInfo(
        id=ModelID.RTM_DET,
        model_class=RTMDetModel,
        variants={
            DEFAULT_KEY: Variant(
                source=UltralyticsSource(
                    format=ModelFormat.ONNX,
                    name="rtmdet",
                ),
                backends={
                    ComputeUnit.CPU: {"model": "model.onnx"},
                    ComputeUnit.GPU: {"model": "model.onnx"},
                },
            ),
            # DLC only — float decode head; v68 NPU context binary not feasible.
            "qcs6490": Variant(
                source=HuggingFaceSource(
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
                backends={
                    ComputeUnit.CPU: {"model": "model.dlc"},
                    ComputeUnit.GPU: {"model": "model.dlc"},
                },
                input_layout="NHWC",
            ),
        },
    ),
    ModelID.DETECTRON2: ModelInfo(
        id=ModelID.DETECTRON2,
        model_class=Detectron2Model,
        variants={
            DEFAULT_KEY: Variant(
                source=HuggingFaceSource(
                    format=ModelFormat.ONNX,
                    hf_repo_id="llamas-lab/m2a-models",
                    hf_subdir="detectron2_float_onnx",
                    files=["model.onnx"],
                    revision="5e24d703d92960e189dfda7173e4dd445ebf2f11",
                ),
                backends={
                    ComputeUnit.CPU: {"model": "model.onnx"},
                    ComputeUnit.GPU: {"model": "model.onnx"},
                },
            ),
            "qcs6490_w8a16": Variant(
                source=HuggingFaceSource(
                    format=ModelFormat.DLC,
                    hf_repo_id="llamas-lab/m2a-models",
                    hf_subdir="detectron2_qcs_w8a16",
                    files=[
                        "model.proposal_generator.npu.bin",
                        "model.roi_head.npu.bin",
                        "metadata.json",
                        "reference_outputs/inputs.npy",
                        "reference_outputs/outputs_0.npy",
                        "reference_outputs/outputs_1.npy",
                        "reference_outputs/outputs_2.npy",
                    ],
                    revision="5e24d703d92960e189dfda7173e4dd445ebf2f11",
                ),
                backends={
                    ComputeUnit.NPU: {
                        "proposal_generator": "model.proposal_generator.npu.bin",
                        "roi_head": "model.roi_head.npu.bin",
                    },
                },
                input_layout="NHWC",
            ),
            "qcs6490_w8a8": Variant(
                source=HuggingFaceSource(
                    format=ModelFormat.DLC,
                    hf_repo_id="llamas-lab/m2a-models",
                    hf_subdir="detectron2_qcs_w8a8",
                    files=[
                        "model.proposal_generator.npu.bin",
                        "model.roi_head.npu.bin",
                        "metadata.json",
                        "reference_outputs/inputs.npy",
                        "reference_outputs/outputs_0.npy",
                        "reference_outputs/outputs_1.npy",
                        "reference_outputs/outputs_2.npy",
                    ],
                    revision="5e24d703d92960e189dfda7173e4dd445ebf2f11",
                ),
                backends={
                    ComputeUnit.NPU: {
                        "proposal_generator": "model.proposal_generator.npu.bin",
                        "roi_head": "model.roi_head.npu.bin",
                    },
                },
                input_layout="NHWC",
            ),
        },
    ),
}
