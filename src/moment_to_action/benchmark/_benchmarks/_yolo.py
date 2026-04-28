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

_IOU_RECALL_THRESHOLD = 0.5
_BBOX_COORDS = 4
_INPUT_NDIM = 4
_CHANNEL_AXIS = 1
_NHWC_CHANNEL_AXIS = 3
_RGB_CHANNELS = 3
_OUTPUT_NDIM = 3
_MATRIX_NDIM = 2
_YOLO_FEATURE_DIM = 84

# Debug/branch constants
_GPU_DEBUG_LOG_IMAGES = 3
_NMS_DIM_2 = 2
_NMS_DIM_3 = 3


class YOLOBenchmark(ModelBenchmark):
    def _log_gpu_debug(
        self,
        idx: int,
        raw_outputs: object,
        gt_det: OracleDetection,
        yolo_boxes: list[OracleBox],
        pre_nms_count: int | None,
        post_nms_count: int,
    ) -> None:
        if idx >= _GPU_DEBUG_LOG_IMAGES:
            return
        # Raw output
        if isinstance(raw_outputs, (list, tuple)):
            arr = raw_outputs[0]
        elif isinstance(raw_outputs, dict):
            arr = next(iter(raw_outputs.values()))
        else:
            arr = raw_outputs
        if isinstance(arr, np.ndarray):
            logger.info(
                "[GPU DEBUG] Raw output shape: %s, dtype: %s, min: %s, max: %s",
                arr.shape,
                arr.dtype,
                arr.min(),
                arr.max(),
            )
            logger.info("[GPU DEBUG] Raw output sample: %s", arr.flatten()[:10])
        # Box counts
        logger.info("[GPU DEBUG] Image: %s", gt_det.image_name)
        logger.info("[GPU DEBUG] Pre-NMS candidate boxes: %s", pre_nms_count)
        logger.info("[GPU DEBUG] Post-NMS boxes: %s", post_nms_count)
        # Predicted boxes
        if yolo_boxes:
            for i, b in enumerate(yolo_boxes[:_GPU_DEBUG_LOG_IMAGES]):
                logger.info(
                    "[GPU DEBUG] Pred box %d: x1=%.1f, y1=%.1f, x2=%.1f, "
                    "y2=%.1f, label=%s, conf=%.3f",
                    i,
                    b.x1,
                    b.y1,
                    b.x2,
                    b.y2,
                    b.label,
                    b.confidence,
                )
        else:
            logger.info("[GPU DEBUG] No predicted boxes after NMS/threshold.")
        # Ground truth boxes
        for i, b in enumerate(gt_det.boxes[:_GPU_DEBUG_LOG_IMAGES]):
            logger.info(
                "[GPU DEBUG] GT box %d: x1=%.1f, y1=%.1f, x2=%.1f, y2=%.1f, label=%s",
                i,
                b.x1,
                b.y1,
                b.x2,
                b.y2,
                b.label,
            )

    def _scale_predicted_boxes(
        self,
        yolo_boxes: list[OracleBox],
        img_path: Path,
    ) -> list[OracleBox]:
        if not yolo_boxes:
            return []
        with Image.open(img_path) as _pil:
            orig_w, orig_h = _pil.size

        def is_normalized(box: OracleBox) -> bool:
            return (
                0.0 <= box.x1 <= 1.0
                and 0.0 <= box.y1 <= 1.0
                and 0.0 <= box.x2 <= 1.0
                and 0.0 <= box.y2 <= 1.0
            )

        if all(is_normalized(b) for b in yolo_boxes):
            return [
                OracleBox(
                    x1=b.x1 * orig_w,
                    y1=b.y1 * orig_h,
                    x2=b.x2 * orig_w,
                    y2=b.y2 * orig_h,
                    label=b.label,
                    confidence=b.confidence,
                )
                for b in yolo_boxes
            ]
        return yolo_boxes

    _input_dtype: type[np.generic]
    """Benchmark implementation for YOLOv12-n ONNX."""

    def __init__(
        self,
        *,
        coco_dataset: CocoDataset | None = None,
        conf_threshold: float = 0.25,
        model_path: str | None = None,
    ) -> None:
        super().__init__()
        self._coco_dataset = coco_dataset
        self._conf_threshold = conf_threshold
        self._model_path = model_path
        self._is_tflite = model_path is not None and str(model_path).endswith(".tflite")
        # Default input shape; will be set in _load_model
        self._input_shape: tuple[int, ...] = (1, 3, 640, 640)

    @property
    def model_id(self) -> ModelID:
        return ModelID.YOLO_V12_N

    def _load_model(self, backend: ComputeBackend, manager: ModelManager) -> object:
        # Detect TFLite by file extension
        model_path = self._model_path or manager.get_path(ModelID.YOLO_V12_N)
        self._is_tflite = str(model_path).endswith(".tflite")
        # INT8 models (QNN delegate) expect uint8 input, float32 for FP32/FP16
        if self._is_tflite:
            if "w8a8" in str(model_path) or "int8" in str(model_path):
                self._input_dtype = np.uint8
            else:
                self._input_dtype = np.float32
            self._input_shape = (1, 640, 640, 3)  # NHWC
        else:
            self._input_dtype = np.float32
            self._input_shape = (1, 3, 640, 640)  # NCHW
        return backend.load_model(model_path)

    def _make_dummy_input(self, handle: object, batch_size: int = 1) -> object:
        del handle
        # Use NHWC for TFLite, NCHW for ONNX
        shape = (batch_size, 640, 640, 3) if self._is_tflite else (batch_size, 3, 640, 640)
        dtype = getattr(self, "_input_dtype", np.float32)
        return np.zeros(shape, dtype=dtype)

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
        del manager
        if self._coco_dataset is None:
            return None
        return self._evaluate_coco_accuracy(handle=handle, backend=backend)

    def _evaluate_coco_accuracy(self, handle: object, backend: ComputeBackend) -> float | None:
        """Evaluate YOLO against native COCO GT detections using mAP@0.50 and mAP@0.75."""
        dataset = self._coco_dataset
        if dataset is None:
            return None

        gt_detections = dataset.instance_detections()
        if not gt_detections:
            logger.debug("YOLOBenchmark: no COCO native detections found -- skipping accuracy.")
            return None

        image_map = {path.name: path for path in dataset.images()}
        predictions = []
        for _idx, gt_det in enumerate(gt_detections):
            img_path = image_map.get(gt_det.image_name)
            if img_path is None:
                continue

            img_tensor = _load_yolo_tensor(img_path, self._input_shape)
            raw_outputs = backend.run(handle, img_tensor)

            # Pre-thresholding: count number of raw candidate boxes (if possible)
            pre_nms_count = None
            if (
                isinstance(raw_outputs, (list, tuple))
                and len(raw_outputs) > 0
                and isinstance(raw_outputs[0], np.ndarray)
            ):
                arr = raw_outputs[0]
                if arr.ndim == _NMS_DIM_3 and arr.shape[0] == 1:
                    arr = arr[0]
                if arr.ndim == _NMS_DIM_2:
                    pre_nms_count = arr.shape[0]
            yolo_boxes = _parse_yolo_boxes(
                raw_outputs,
                self._input_shape,
                conf_threshold=self._conf_threshold,
                class_labels=YOLOStage.COCO_LABELS,
            )
            post_nms_count = len(yolo_boxes)
            self._log_gpu_debug(
                _idx,
                raw_outputs,
                gt_det,
                yolo_boxes,
                pre_nms_count,
                post_nms_count,
            )
            scaled_boxes = self._scale_predicted_boxes(yolo_boxes, img_path)
            predictions.append(OracleDetection(image_name=gt_det.image_name, boxes=scaled_boxes))

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
            "YOLOBenchmark COCO native GT: mAP@0.5=%.3f mAP@0.75=%.3f recall@0.5=%.3f",
            metrics.map_50,
            metrics.map_75,
            metrics.recall_50,
        )
        return metrics.map_50


def _load_yolo_tensor(
    img_path: Path,
    input_shape: tuple[int, ...],
) -> np.ndarray:
    """Load a PIL image and convert to a float32 tensor matching input_shape."""
    # Detect NHWC vs NCHW by input_shape
    # INT8 TFLite expects uint8 [0,255], float models expect float32 [0,1]
    is_nhwc = len(input_shape) == _INPUT_NDIM and input_shape[_NHWC_CHANNEL_AXIS] == _RGB_CHANNELS
    is_nchw = len(input_shape) == _INPUT_NDIM and input_shape[_CHANNEL_AXIS] == _RGB_CHANNELS
    if is_nchw:
        _, _, height, width = input_shape
        image = Image.open(img_path).convert("RGB")
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        return arr[np.newaxis]
    if is_nhwc:
        _, height, width, _ = input_shape
        image = Image.open(img_path).convert("RGB")
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32)
        # Always return float32 in [0,1] here; let the caller cast to uint8 if needed
        arr = arr / 255.0
        return arr[np.newaxis]
    msg = f"Unsupported input shape for YOLO tensor: {input_shape}"
    raise ValueError(msg)


def _letterbox_resize(
    img_bgr: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Resize an image with letterboxing while preserving aspect ratio."""
    src_h, src_w = img_bgr.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    import cv2  # type: ignore[import-untyped]

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas


def _preprocess_nchw(img_bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return a float32 RGB NCHW tensor normalized to [0, 1]."""
    canvas = _letterbox_resize(img_bgr, height, width)
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return np.expand_dims(rgb.transpose(2, 0, 1), 0)


def _preprocess_nhwc(img_bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return a float32 RGB NHWC tensor normalized to [0, 1]."""
    canvas = _letterbox_resize(img_bgr, height, width)
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return np.expand_dims(rgb, 0)


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    """Pure-numpy greedy NMS. Returns indices to keep, sorted by descending score."""
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while len(order) > 0:
        cur = order[0]
        keep.append(int(cur))
        if len(order) == 1:
            break
        cb = boxes[cur]
        rb = boxes[order[1:]]
        x1 = np.maximum(cb[0], rb[:, 0])
        y1 = np.maximum(cb[1], rb[:, 1])
        x2 = np.minimum(cb[2], rb[:, 2])
        y2 = np.minimum(cb[3], rb[:, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        cur_area = (cb[2] - cb[0]) * (cb[3] - cb[1])
        rem_areas = (rb[:, 2] - rb[:, 0]) * (rb[:, 3] - rb[:, 1])
        iou = inter / (cur_area + rem_areas - inter + 1e-6)
        order = order[1:][iou < iou_threshold]
    return keep


def _build_oracle_boxes(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    img_w: int,
    img_h: int,
    class_labels: tuple[str, ...] | None,
) -> list[OracleBox]:
    """Convert filtered xyxy arrays to OracleBox instances, clamped to image bounds."""
    results: list[OracleBox] = []
    for box, conf, cls_id in zip(boxes_xyxy, scores, class_ids, strict=False):
        x1 = max(0.0, float(box[0]))
        y1 = max(0.0, float(box[1]))
        x2 = min(float(img_w), float(box[2]))
        y2 = min(float(img_h), float(box[3]))
        class_name = str(int(cls_id))
        if class_labels is not None and int(cls_id) < len(class_labels):
            class_name = class_labels[int(cls_id)]
        results.append(
            OracleBox(x1=x1, y1=y1, x2=x2, y2=y2, label=class_name, confidence=float(conf))
        )
    return results


def _parse_yolo_boxes(  # noqa: C901, PLR0911, PLR0912
    raw_outputs: object,
    input_shape: tuple[int, ...],
    conf_threshold: float = 0.25,
    class_labels: tuple[str, ...] | None = None,
) -> list[OracleBox]:
    """Parse YOLO raw output tensors into OracleBox instances.

    Handles two output formats:

    * **3-tensor** ``[boxes(1,N,4), scores(1,N), class_ids(1,N)]`` — xyxy pixel coords
      in model input space (e.g. 640x640).  Produced by the vendored ONNX model.
    * **1-tensor combined** ``(1, 84, N)`` — cx/cy/w/h + 80 class scores.
      Produced by the standard Ultralytics TFLite export.
    """
    if len(input_shape) == _INPUT_NDIM and input_shape[_CHANNEL_AXIS] == _RGB_CHANNELS:
        _, _, img_h, img_w = input_shape
    else:
        _, img_h, img_w, _ = input_shape

    # --- 3-tensor format ---
    if (
        isinstance(raw_outputs, (list, tuple))
        and len(raw_outputs) == _OUTPUT_NDIM  # 3 tensors
        and all(isinstance(t, np.ndarray) for t in raw_outputs)
    ):
        boxes_arr, scores_arr, class_ids_arr = raw_outputs[0], raw_outputs[1], raw_outputs[2]
        if boxes_arr.ndim == _OUTPUT_NDIM and boxes_arr.shape[-1] == _BBOX_COORDS:  # (1,N,4)
            boxes_xyxy = boxes_arr[0].astype(np.float32)
            scores = scores_arr[0].astype(np.float32)
            class_ids: np.ndarray = class_ids_arr[0]

            mask = scores >= conf_threshold
            boxes_xyxy, scores, class_ids = boxes_xyxy[mask], scores[mask], class_ids[mask]
            if len(boxes_xyxy) == 0:
                return []

            keep = _nms_numpy(boxes_xyxy, scores)
            return _build_oracle_boxes(
                boxes_xyxy[keep], scores[keep], class_ids[keep], img_w, img_h, class_labels
            )

    # --- 1-tensor / combined format ---
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

    if arr.ndim != _MATRIX_NDIM:
        return []

    if arr.shape[0] == _YOLO_FEATURE_DIM:
        arr = arr.T
    if arr.shape[0] == 0 or arr.shape[1] <= _BBOX_COORDS:
        return []

    boxes_xywh = arr[:, :_BBOX_COORDS]
    class_scores = arr[:, _BBOX_COORDS:]
    confidences = class_scores.max(axis=1)

    mask = confidences >= conf_threshold
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids_raw: np.ndarray = class_scores[mask].argmax(axis=1)

    if len(boxes_xywh) == 0:
        return []

    x1s = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1s = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2s = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2s = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy_raw = np.stack([x1s, y1s, x2s, y2s], axis=1)

    keep = _nms_numpy(boxes_xyxy_raw, confidences)
    return _build_oracle_boxes(
        boxes_xyxy_raw[keep], confidences[keep], class_ids_raw[keep], img_w, img_h, class_labels
    )
