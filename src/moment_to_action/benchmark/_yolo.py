from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from moment_to_action.benchmark._base import ModelBenchmark
from moment_to_action.benchmark._detection_metrics import compute_detection_map
from moment_to_action.benchmark._oracle_ground_truth import OracleBox, OracleDetection, OracleStore
from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models import ModelID
from moment_to_action.stages.video._yolo import YOLOStage

if TYPE_CHECKING:
    from moment_to_action.benchmark._coco_dataset import CocoDataset
    from moment_to_action.hardware import ComputeBackend
    from moment_to_action.models import ModelManager

logger = logging.getLogger(__name__)

_IOU_RECALL_THRESHOLD = 0.5
_INPUT_NDIM = 4
_CHANNEL_AXIS = 1
_RGB_CHANNELS = 3
_OUTPUT_NDIM = 3
_MATRIX_NDIM = 2
_YOLO_FEATURE_DIM = 84


class YOLOBenchmark(ModelBenchmark):
    """Benchmark implementation for YOLOv8.

    Loads the TFLite variant (``YOLO_V8_TFLITE``) on accelerated compute units
    so inference runs through the LiteRT/QNN path instead of ONNX/CPU. Falls
    back to the ONNX variant when the TFLite model has not yet been converted or
    when the active unit is CPU.
    """

    def __init__(
        self,
        *,
        coco_dataset: CocoDataset | None = None,
        conf_threshold: float = 0.25,
    ) -> None:
        super().__init__()
        self._input_shape: tuple[int, ...] = (1, 3, 640, 640)
        self._coco_dataset = coco_dataset
        self._conf_threshold = conf_threshold

    @property
    def model_id(self) -> ModelID:
        return ModelID.YOLO_V8

    def _load_model(self, backend: ComputeBackend, manager: ModelManager) -> object:
        if backend.active_unit == ComputeUnit.NPU:
            for npu_model_id in (ModelID.YOLO_V8_TFLITE_INT8_320, ModelID.YOLO_V8_TFLITE_INT8):
                if manager.is_available(npu_model_id):
                    handle = backend.load_model(manager.get_path(npu_model_id))
                    details = backend.get_input_details(handle)
                    self._input_shape = tuple(int(dimension) for dimension in details[0]["shape"])
                    return handle

        if backend.active_unit != ComputeUnit.CPU and manager.is_available(ModelID.YOLO_V8_TFLITE):
            handle = backend.load_model(manager.get_path(ModelID.YOLO_V8_TFLITE))
            details = backend.get_input_details(handle)
            self._input_shape = tuple(int(dimension) for dimension in details[0]["shape"])
            return handle

        self._input_shape = (1, 3, 640, 640)
        return backend.load_model(manager.get_path(ModelID.YOLO_V8))

    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object:
        del handle
        shape = (batch_size, *self._input_shape[1:])
        return np.zeros(shape, dtype=np.float32)

    def _run_inference(self, handle: object, inputs: object, backend: ComputeBackend) -> None:
        if not isinstance(inputs, np.ndarray):
            msg = "YOLO benchmark expects ndarray inputs"
            raise TypeError(msg)
        backend.run(handle, inputs)

    def _evaluate_accuracy(
        self,
        handle: object,
        backend: ComputeBackend,
        manager: ModelManager,
    ) -> float | None:
        """Evaluate YOLO against project oracle or COCO pseudo-ground-truth."""
        if self._coco_dataset is not None:
            return self._evaluate_coco_accuracy(handle=handle, backend=backend)

        del manager
        gt = OracleStore().load()
        if gt is None or not gt.detections:
            logger.debug("YOLOBenchmark: no oracle detections found -- skipping accuracy.")
            return None

        sample_images = _default_sample_images()
        if not sample_images:
            return None

        image_map = {path.name: path for path in sample_images}
        matched = 0
        total = 0

        for oracle_det in gt.detections:
            img_path = image_map.get(oracle_det.image_name)
            if img_path is None or not oracle_det.boxes:
                continue

            img_tensor = _load_yolo_tensor(img_path, self._input_shape)
            raw_outputs = backend.run(handle, img_tensor)

            yolo_boxes = _parse_yolo_boxes(raw_outputs, self._input_shape)
            for oracle_box in oracle_det.boxes:
                total += 1
                best_iou = max((oracle_box.iou(box) for box in yolo_boxes), default=0.0)
                if best_iou >= _IOU_RECALL_THRESHOLD:
                    matched += 1

        if total == 0:
            return None

        recall = matched / total
        self._set_accuracy_details({"recall_50": recall})
        logger.info("YOLOBenchmark accuracy: recall@IoU0.5 = %.3f (%d/%d)", recall, matched, total)
        return recall

    def _evaluate_coco_accuracy(self, handle: object, backend: ComputeBackend) -> float | None:
        """Evaluate YOLO against COCO oracle pseudo-labels using mAP@[0.5:0.95]."""
        dataset = self._coco_dataset
        if dataset is None:
            return None

        gt = OracleStore(dataset_name=dataset.dataset_name).load()
        if gt is None or not gt.detections:
            logger.debug("YOLOBenchmark: no COCO oracle detections found -- skipping accuracy.")
            return None

        image_map = {path.name: path for path in dataset.images()}
        predictions = []
        for oracle_det in gt.detections:
            img_path = image_map.get(oracle_det.image_name)
            if img_path is None:
                continue

            img_tensor = _load_yolo_tensor(img_path, self._input_shape)
            raw_outputs = backend.run(handle, img_tensor)
            yolo_boxes = _parse_yolo_boxes(
                raw_outputs,
                self._input_shape,
                conf_threshold=self._conf_threshold,
                class_labels=YOLOStage.COCO_LABELS,
            )
            predictions.append(OracleDetection(image_name=oracle_det.image_name, boxes=yolo_boxes))

        if not predictions:
            return None

        metrics = compute_detection_map(predictions=predictions, ground_truth=gt.detections)
        self._set_accuracy_details(
            {
                "map_50": metrics.map_50,
                "map_50_95": metrics.map_50_95,
                "recall_50": metrics.recall_50,
            }
        )
        logger.info(
            "YOLOBenchmark COCO pseudo-GT: mAP@[0.5:0.95]=%.3f mAP@0.5=%.3f recall@0.5=%.3f",
            metrics.map_50_95,
            metrics.map_50,
            metrics.recall_50,
        )
        return metrics.map_50_95


def _load_yolo_tensor(
    img_path: Path,
    input_shape: tuple[int, ...],
) -> np.ndarray:
    """Load a PIL image and convert to a float32 tensor matching input_shape."""
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


def _parse_yolo_boxes(
    raw_outputs: object,
    input_shape: tuple[int, ...],
    conf_threshold: float = 0.25,
    class_labels: tuple[str, ...] | None = None,
) -> list[OracleBox]:
    """Parse YOLO raw output tensors into OracleBox instances."""
    if isinstance(raw_outputs, (list, tuple)):
        arr = raw_outputs[0]
    elif isinstance(raw_outputs, dict):
        arr = next(iter(raw_outputs.values()))
    else:
        arr = raw_outputs

    if not isinstance(arr, np.ndarray):
        return []

    if arr.ndim == _OUTPUT_NDIM and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == _MATRIX_NDIM:
        if arr.shape[0] == _YOLO_FEATURE_DIM:
            arr = arr.T
        boxes_xywh = arr[:, :4]
        class_scores = arr[:, 4:]
        confidences = class_scores.max(axis=1)

        mask = confidences >= conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_scores[mask].argmax(axis=1)
    else:
        return []

    if len(input_shape) == _INPUT_NDIM and input_shape[_CHANNEL_AXIS] == _RGB_CHANNELS:
        _, _, img_h, img_w = input_shape
    else:
        _, img_h, img_w, _ = input_shape

    results: list[OracleBox] = []
    for (cx, cy, box_w, box_h), conf, cls_id in zip(
        boxes_xywh,
        confidences,
        class_ids,
        strict=False,
    ):
        x1 = float(cx - box_w / 2)
        y1 = float(cy - box_h / 2)
        x2 = float(cx + box_w / 2)
        y2 = float(cy + box_h / 2)
        x1, x2 = max(0.0, x1), min(float(img_w), x2)
        y1, y2 = max(0.0, y1), min(float(img_h), y2)
        class_name = str(int(cls_id))
        if class_labels is not None and int(cls_id) < len(class_labels):
            class_name = class_labels[int(cls_id)]

        results.append(
            OracleBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                label=class_name,
                confidence=float(conf),
            )
        )
    return results


def _default_sample_images() -> list[Path]:
    """Locate the project images/ directory relative to this file."""
    candidate = Path(__file__).parents[4] / "images"
    if candidate.is_dir():
        return sorted(candidate.glob("*.jpg"))
    return []
