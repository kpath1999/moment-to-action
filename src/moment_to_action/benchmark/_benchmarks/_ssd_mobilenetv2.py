from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from moment_to_action.benchmark._benchmarks._base import ModelBenchmark
from moment_to_action.benchmark._detection_metrics import compute_detection_map
from moment_to_action.benchmark._oracle_ground_truth import OracleBox, OracleDetection
from moment_to_action.models import ModelID

if TYPE_CHECKING:
    from pathlib import Path

    from moment_to_action.benchmark._datasets._coco_dataset import CocoDataset
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager

logger = logging.getLogger(__name__)

_INPUT_NDIM = 4
_CHANNEL_AXIS = 1
_RGB_CHANNELS = 3
_OUTPUT_NDIM = 3
_CLASS_SCORES_NDIM = 2
_BOX_COORDS = 4
_MIN_OUTPUT_TENSORS = 3
_WITH_COUNT_TENSORS = 4

_COCO_LABELS_BY_ID: dict[int, str] = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


class SSDMobileNetV2Benchmark(ModelBenchmark):
    """Benchmark implementation for SSD-MobileNet-v2."""

    def __init__(
        self,
        *,
        coco_dataset: CocoDataset | None = None,
        conf_threshold: float = 0.25,
    ) -> None:
        super().__init__()
        self._coco_dataset = coco_dataset
        self._conf_threshold = conf_threshold
        self._input_shape: tuple[int, ...] = (1, 3, 300, 300)

    @property
    def model_id(self) -> ModelID:
        return ModelID.SSD_MOBILENETV2

    def _load_model(self, backend: ComputeBackend, manager: ModelManager) -> object:
        handle = backend.load_model(manager.get_path(self.model_id))
        details = backend.get_input_details(handle)
        self._input_shape = tuple(int(dimension) for dimension in details[0]["shape"])
        return handle

    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object:
        del handle
        shape = (batch_size, *self._input_shape[1:])
        return np.zeros(shape, dtype=np.float32)

    def _run_inference(self, handle: object, inputs: object, backend: ComputeBackend) -> None:
        if not isinstance(inputs, np.ndarray):
            msg = "SSDMobileNetV2Benchmark expects ndarray inputs"
            raise TypeError(msg)
        backend.run(handle, inputs)

    def _evaluate_accuracy(
        self,
        handle: object,
        backend: ComputeBackend,
        manager: ModelManager,
    ) -> float | None:
        del manager
        if self._coco_dataset is None:
            return None
        return self._evaluate_coco_accuracy(handle=handle, backend=backend)

    def _evaluate_coco_accuracy(self, handle: object, backend: ComputeBackend) -> float | None:
        dataset = self._coco_dataset
        if dataset is None:
            return None

        gt_detections = dataset.instance_detections()
        if not gt_detections:
            return None

        image_map = {path.name: path for path in dataset.images()}
        predictions: list[OracleDetection] = []
        for gt_det in gt_detections:
            img_path = image_map.get(gt_det.image_name)
            if img_path is None:
                continue

            img_tensor = _load_ssd_tensor(img_path, self._input_shape)
            raw_outputs = backend.run(handle, img_tensor)
            with Image.open(img_path) as image:
                orig_w, orig_h = image.size

            boxes = _parse_ssd_boxes(
                raw_outputs,
                image_width=orig_w,
                image_height=orig_h,
                conf_threshold=self._conf_threshold,
            )
            predictions.append(OracleDetection(image_name=gt_det.image_name, boxes=boxes))

        if not predictions:
            return None

        metrics = compute_detection_map(predictions=predictions, ground_truth=gt_detections)
        self._set_accuracy_details(
            {
                "map_50": metrics.map_50,
                "map_75": metrics.map_75,
                "recall_50": metrics.recall_50,
            }
        )
        logger.info(
            "SSDMobileNetV2Benchmark COCO native GT: mAP@0.5=%.3f mAP@0.75=%.3f recall@0.5=%.3f",
            metrics.map_50,
            metrics.map_75,
            metrics.recall_50,
        )
        return metrics.map_50


def _load_ssd_tensor(img_path: Path, input_shape: tuple[int, ...]) -> np.ndarray:
    """Load an image and format it as float32 matching the model input layout."""
    if len(input_shape) == _INPUT_NDIM and input_shape[_CHANNEL_AXIS] == _RGB_CHANNELS:
        _, _, height, width = input_shape
        nchw = True
    else:
        _, height, width, _ = input_shape
        nchw = False

    image = Image.open(img_path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if nchw:
        arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis]


def _parse_ssd_boxes(
    raw_outputs: object,
    *,
    image_width: int,
    image_height: int,
    conf_threshold: float = 0.25,
) -> list[OracleBox]:
    """Parse SSD outputs into OracleBox values.

    Expected output tensors:
    - detection_boxes: [1, N, 4] in normalized [ymin, xmin, ymax, xmax]
    - detection_classes: [1, N]
    - detection_scores: [1, N]
    - num_detections: [1]
    """
    boxes_arr: np.ndarray | None = None
    classes_arr: np.ndarray | None = None
    scores_arr: np.ndarray | None = None
    count_arr: np.ndarray | None = None

    if isinstance(raw_outputs, dict):
        boxes_arr = _as_array(raw_outputs.get("detection_boxes"))
        classes_arr = _as_array(raw_outputs.get("detection_classes"))
        scores_arr = _as_array(raw_outputs.get("detection_scores"))
        count_arr = _as_array(raw_outputs.get("num_detections"))

    if (
        (boxes_arr is None or classes_arr is None or scores_arr is None)
        and isinstance(raw_outputs, (list, tuple))
        and len(raw_outputs) >= _MIN_OUTPUT_TENSORS
    ):
        boxes_arr = _as_array(raw_outputs[0])
        classes_arr = _as_array(raw_outputs[1])
        scores_arr = _as_array(raw_outputs[2])
        if len(raw_outputs) >= _WITH_COUNT_TENSORS:
            count_arr = _as_array(raw_outputs[3])

    if boxes_arr is None or classes_arr is None or scores_arr is None:
        return []

    if (
        boxes_arr.ndim != _OUTPUT_NDIM
        or boxes_arr.shape[0] != 1
        or boxes_arr.shape[-1] != _BOX_COORDS
    ):
        return []
    if classes_arr.ndim != _CLASS_SCORES_NDIM or scores_arr.ndim != _CLASS_SCORES_NDIM:
        return []

    boxes = boxes_arr[0]
    classes = classes_arr[0].astype(np.int32)
    scores = scores_arr[0].astype(np.float32)

    max_count = min(len(boxes), len(classes), len(scores))
    if count_arr is not None and count_arr.size > 0:
        max_count = min(max_count, int(count_arr.flat[0]))

    results: list[OracleBox] = []
    for idx in range(max_count):
        score = float(scores[idx])
        if score < conf_threshold:
            continue

        ymin, xmin, ymax, xmax = boxes[idx].tolist()
        x1 = max(0.0, float(xmin) * image_width)
        y1 = max(0.0, float(ymin) * image_height)
        x2 = min(float(image_width), float(xmax) * image_width)
        y2 = min(float(image_height), float(ymax) * image_height)

        class_id = int(classes[idx])
        label = _COCO_LABELS_BY_ID.get(class_id, str(class_id))
        results.append(OracleBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label, confidence=score))
    return results


def _as_array(value: object) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    return None
