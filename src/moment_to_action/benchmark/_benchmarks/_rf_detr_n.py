from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from moment_to_action.benchmark._benchmarks._base import ModelBenchmark
from moment_to_action.benchmark._detection_metrics import compute_detection_map
from moment_to_action.benchmark._oracle_ground_truth import OracleBox, OracleDetection
from moment_to_action.models import ModelID
from moment_to_action.stages.video._yolo import YOLOStage

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
_BOX_COORDS = 4
_MIN_OUTPUT_TENSORS = 2


class RFDETRBenchmark(ModelBenchmark):
    """Benchmark implementation for RF-DETR-n ONNX variants."""

    def __init__(
        self,
        *,
        coco_dataset: CocoDataset | None = None,
        conf_threshold: float = 0.25,
    ) -> None:
        super().__init__()
        self._coco_dataset = coco_dataset
        self._conf_threshold = conf_threshold
        self._input_shape: tuple[int, ...] = (1, 3, 640, 640)

    @property
    def model_id(self) -> ModelID:
        return ModelID.RF_DETR_N

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
            msg = "RFDETRBenchmark expects ndarray inputs"
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

            img_tensor = _load_rfdetr_tensor(img_path, self._input_shape)
            raw_outputs = backend.run(handle, img_tensor)
            with Image.open(img_path) as image:
                orig_w, orig_h = image.size

            boxes = _parse_rfdetr_boxes(
                raw_outputs,
                image_width=orig_w,
                image_height=orig_h,
                conf_threshold=self._conf_threshold,
                class_labels=YOLOStage.COCO_LABELS,
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
            "RFDETRBenchmark COCO native GT: mAP@0.5=%.3f mAP@0.75=%.3f recall@0.5=%.3f",
            metrics.map_50,
            metrics.map_75,
            metrics.recall_50,
        )
        return metrics.map_50


def _load_rfdetr_tensor(img_path: Path, input_shape: tuple[int, ...]) -> np.ndarray:
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


def _parse_rfdetr_boxes(  # noqa: C901, PLR0912
    raw_outputs: object,
    *,
    image_width: int,
    image_height: int,
    conf_threshold: float = 0.25,
    class_labels: tuple[str, ...] | None = None,
) -> list[OracleBox]:
    """Parse RF-DETR outputs into OracleBox values.

    Expected output tensors:
    - logits: [1, num_queries, num_classes]
    - pred_boxes: [1, num_queries, 4] normalized cxcywh
    """
    logits_arr: np.ndarray | None = None
    boxes_arr: np.ndarray | None = None

    if isinstance(raw_outputs, dict):
        for value in raw_outputs.values():
            arr = _as_array(value)
            if arr is None:
                continue
            if arr.ndim == _OUTPUT_NDIM and arr.shape[-1] == _BOX_COORDS:
                boxes_arr = arr
            elif arr.ndim == _OUTPUT_NDIM:
                logits_arr = arr

    if (
        (logits_arr is None or boxes_arr is None)
        and isinstance(raw_outputs, (list, tuple))
        and len(raw_outputs) >= _MIN_OUTPUT_TENSORS
    ):
        arrays = [arr for item in raw_outputs if (arr := _as_array(item)) is not None]
        if len(arrays) >= _MIN_OUTPUT_TENSORS:
            if arrays[0].shape[-1] == _BOX_COORDS:
                boxes_arr, logits_arr = arrays[0], arrays[1]
            else:
                logits_arr, boxes_arr = arrays[0], arrays[1]

    if logits_arr is None or boxes_arr is None:
        return []

    if logits_arr.ndim != _OUTPUT_NDIM or boxes_arr.ndim != _OUTPUT_NDIM:
        return []
    if logits_arr.shape[0] != 1 or boxes_arr.shape[0] != 1 or boxes_arr.shape[-1] != _BOX_COORDS:
        return []

    logits = logits_arr[0].astype(np.float32)
    boxes_xywh = boxes_arr[0].astype(np.float32)
    if logits.shape[0] != boxes_xywh.shape[0]:
        return []

    probs = 1.0 / (1.0 + np.exp(-logits))
    confidences = probs.max(axis=1)
    class_ids = probs.argmax(axis=1)

    mask = confidences >= conf_threshold
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    cx = boxes_xywh[:, 0] * image_width
    cy = boxes_xywh[:, 1] * image_height
    width = boxes_xywh[:, 2] * image_width
    height = boxes_xywh[:, 3] * image_height

    x1 = np.clip(cx - width / 2.0, 0.0, image_width)
    y1 = np.clip(cy - height / 2.0, 0.0, image_height)
    x2 = np.clip(cx + width / 2.0, 0.0, image_width)
    y2 = np.clip(cy + height / 2.0, 0.0, image_height)

    results: list[OracleBox] = []
    for bx1, by1, bx2, by2, class_id, confidence in zip(
        x1,
        y1,
        x2,
        y2,
        class_ids,
        confidences,
        strict=False,
    ):
        class_name = str(int(class_id))
        if class_labels is not None and int(class_id) < len(class_labels):
            class_name = class_labels[int(class_id)]
        results.append(
            OracleBox(
                x1=float(bx1),
                y1=float(by1),
                x2=float(bx2),
                y2=float(by2),
                label=class_name,
                confidence=float(confidence),
            )
        )
    return results


def _as_array(value: object) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    return None
