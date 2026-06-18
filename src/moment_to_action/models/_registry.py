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
from .llm.phi35._model import Phi35Model
from .llm.qwen2._model import Qwen2Model
from .llm.qwen3._model import Qwen3Model
from .vlm.ministral._model import MinistralVLModel
from .vlm.qwen3_vl._model import Qwen3VLModel
from .vlm.qwen25_vl._model import Qwen25VLModel

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
    ModelID.QWEN2_1_5B_INSTRUCT: ModelInfo(
        id=ModelID.QWEN2_1_5B_INSTRUCT,
        model_class=Qwen2Model,
        variants={
            DEFAULT_KEY: Variant(
                source=HuggingFaceSource(
                    format=ModelFormat.GGUF,
                    hf_repo_id="Qwen/Qwen2-1.5B-Instruct-GGUF",
                    files=["qwen2-1_5b-instruct-q4_0.gguf"],
                    revision="c62434db644497c0ee545c690bb66a67eba6eb3f",
                ),
                backends={ComputeUnit.GPU: {"model": "qwen2-1_5b-instruct-q4_0.gguf"}},
                input_layout=None,
            ),
        },
    ),
    ModelID.QWEN2_7B_INSTRUCT: ModelInfo(
        id=ModelID.QWEN2_7B_INSTRUCT,
        model_class=Qwen2Model,
        variants={
            DEFAULT_KEY: Variant(
                source=HuggingFaceSource(
                    format=ModelFormat.GGUF,
                    hf_repo_id="Qwen/Qwen2-7B-Instruct-GGUF",
                    files=["qwen2-7b-instruct-q4_k_m.gguf"],
                    revision="c3024c6fff0a02d52119ecee024bbb93d4b4b8e4",
                ),
                backends={ComputeUnit.GPU: {"model": "qwen2-7b-instruct-q4_k_m.gguf"}},
                input_layout=None,
            ),
        },
    ),
    ModelID.QWEN3_4B: ModelInfo(
        id=ModelID.QWEN3_4B,
        model_class=Qwen3Model,
        variants={
            DEFAULT_KEY: Variant(
                source=HuggingFaceSource(
                    format=ModelFormat.GGUF,
                    hf_repo_id="Qwen/Qwen3-4B-GGUF",
                    files=["Qwen3-4B-Q4_K_M.gguf"],
                    revision="bc640142c66e1fdd12af0bd68f40445458f3869b",
                ),
                backends={ComputeUnit.GPU: {"model": "Qwen3-4B-Q4_K_M.gguf"}},
                input_layout=None,
            ),
        },
    ),
    ModelID.PHI35_MINI_INSTRUCT: ModelInfo(
        id=ModelID.PHI35_MINI_INSTRUCT,
        model_class=Phi35Model,
        variants={
            DEFAULT_KEY: Variant(
                source=HuggingFaceSource(
                    format=ModelFormat.GGUF,
                    hf_repo_id="bartowski/Phi-3.5-mini-instruct-GGUF",
                    files=["Phi-3.5-mini-instruct-Q4_0.gguf"],
                    revision="6d70da17e749a471ccb62ade694486011a75cda3",
                ),
                backends={ComputeUnit.GPU: {"model": "Phi-3.5-mini-instruct-Q4_0.gguf"}},
                input_layout=None,
            ),
        },
    ),
    ModelID.QWEN25_VL_3B_INSTRUCT: ModelInfo(
        id=ModelID.QWEN25_VL_3B_INSTRUCT,
        model_class=Qwen25VLModel,
        variants={
            DEFAULT_KEY: Variant(
                source=HuggingFaceSource(
                    format=ModelFormat.GGUF,
                    hf_repo_id="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
                    files=[
                        "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
                        "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
                    ],
                    revision="5037fcf163dd95d1e41d1974465f0898ed108ca2",
                ),
                backends={
                    ComputeUnit.GPU: {
                        "model": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
                        "mmproj": "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
                    }
                },
                input_layout=None,
            ),
        },
    ),
    ModelID.QWEN25_VL_7B_INSTRUCT: ModelInfo(
        id=ModelID.QWEN25_VL_7B_INSTRUCT,
        model_class=Qwen25VLModel,
        variants={
            DEFAULT_KEY: Variant(
                source=HuggingFaceSource(
                    format=ModelFormat.GGUF,
                    hf_repo_id="ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
                    files=[
                        "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                        "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf",
                    ],
                    revision="508edd0afaa66bb9e9f40587acc2184f02daf1f6",
                ),
                backends={
                    ComputeUnit.GPU: {
                        "model": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                        "mmproj": "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf",
                    }
                },
                input_layout=None,
            ),
        },
    ),
    ModelID.QWEN3_VL_2B_INSTRUCT: ModelInfo(
        id=ModelID.QWEN3_VL_2B_INSTRUCT,
        model_class=Qwen3VLModel,
        variants={
            DEFAULT_KEY: Variant(
                source=HuggingFaceSource(
                    format=ModelFormat.GGUF,
                    hf_repo_id="Qwen/Qwen3-VL-2B-Instruct-GGUF",
                    files=[
                        "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
                        "mmproj-Qwen3VL-2B-Instruct-F16.gguf",
                    ],
                    revision="52d6c8ffea26cc873ac5ad116f8631268d7eb503",
                ),
                backends={
                    ComputeUnit.GPU: {
                        "model": "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
                        "mmproj": "mmproj-Qwen3VL-2B-Instruct-F16.gguf",
                    }
                },
                input_layout=None,
            ),
        },
    ),
    ModelID.QWEN3_VL_4B_INSTRUCT: ModelInfo(
        id=ModelID.QWEN3_VL_4B_INSTRUCT,
        model_class=Qwen3VLModel,
        variants={
            DEFAULT_KEY: Variant(
                source=HuggingFaceSource(
                    format=ModelFormat.GGUF,
                    hf_repo_id="lmstudio-community/Qwen3-VL-4B-Instruct-GGUF",
                    files=[
                        "Qwen3-VL-4B-Instruct-Q4_K_M.gguf",
                        "mmproj-Qwen3-VL-4B-Instruct-F16.gguf",
                    ],
                    revision="9eaf9988fe9b5e33541dc614622c36d5e90dd509",
                ),
                backends={
                    ComputeUnit.GPU: {
                        "model": "Qwen3-VL-4B-Instruct-Q4_K_M.gguf",
                        "mmproj": "mmproj-Qwen3-VL-4B-Instruct-F16.gguf",
                    }
                },
                input_layout=None,
            ),
        },
    ),
    ModelID.MINISTRAL_3B_INSTRUCT: ModelInfo(
        id=ModelID.MINISTRAL_3B_INSTRUCT,
        model_class=MinistralVLModel,
        variants={
            DEFAULT_KEY: Variant(
                source=HuggingFaceSource(
                    format=ModelFormat.GGUF,
                    hf_repo_id="ggml-org/Ministral-3-3B-Instruct-2512-GGUF",
                    files=[
                        "Ministral-3-3B-Instruct-2512-Q8_0.gguf",
                        "mmproj-Ministral-3-3B-Instruct-2512-Q8_0.gguf",
                    ],
                    revision="742ab8db17d5c8ee5dc8f5afb5acfc2da1c33b26",
                ),
                backends={
                    ComputeUnit.GPU: {
                        "model": "Ministral-3-3B-Instruct-2512-Q8_0.gguf",
                        "mmproj": "mmproj-Ministral-3-3B-Instruct-2512-Q8_0.gguf",
                    }
                },
                input_layout=None,
            ),
        },
    ),
}
