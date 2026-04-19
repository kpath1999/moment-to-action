from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from moment_to_action.benchmark._accuracy import compute_map50, parse_yolo_outputs
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
_BBOX_COORDS = 4
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
        eval_image_paths: list[Path] | None = None,
        *,
        coco_dataset: CocoDataset | None = None,
        conf_threshold: float = 0.25,
        oracle_store: OracleStore | None = None,
    ) -> None:
        super().__init__()
        self._eval_image_paths = eval_image_paths or []
        self._input_shape: tuple[int, ...] = (1, 3, 640, 640)
        self._coco_dataset = coco_dataset
        self._conf_threshold = conf_threshold
        self._oracle_store = oracle_store

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

        if self._eval_image_paths:
            return self._evaluate_eval_images(handle=handle, backend=backend, manager=manager)

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
        """Evaluate YOLO against native COCO GT detections using mAP@[0.5:0.95]."""
        dataset = self._coco_dataset
        if dataset is None:
            return None

        gt_detections = dataset.instance_detections()
        if not gt_detections:
            logger.debug("YOLOBenchmark: no COCO native detections found -- skipping accuracy.")
            return None

        image_map = {path.name: path for path in dataset.images()}
        predictions = []
        for gt_det in gt_detections:
            img_path = image_map.get(gt_det.image_name)
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
            # Scale boxes from model input space (e.g. 640x640) to original image space
            # so they align with COCO ground-truth coordinates.
            if yolo_boxes:
                with Image.open(img_path) as _pil:
                    orig_w, orig_h = _pil.size
                model_h = (
                    self._input_shape[2]
                    if self._input_shape[_CHANNEL_AXIS] == _RGB_CHANNELS
                    else self._input_shape[1]
                )
                model_w = (
                    self._input_shape[3]
                    if self._input_shape[_CHANNEL_AXIS] == _RGB_CHANNELS
                    else self._input_shape[2]
                )
                sx = orig_w / model_w
                sy = orig_h / model_h
                scaled_boxes = [
                    OracleBox(
                        x1=b.x1 * sx,
                        y1=b.y1 * sy,
                        x2=b.x2 * sx,
                        y2=b.y2 * sy,
                        label=b.label,
                        confidence=b.confidence,
                    )
                    for b in yolo_boxes
                ]
            else:
                scaled_boxes = []
            predictions.append(OracleDetection(image_name=gt_det.image_name, boxes=scaled_boxes))

        if not predictions:
            return None

        metrics = compute_detection_map(predictions=predictions, ground_truth=gt_detections)
        self._set_accuracy_details(
            {
                "map_50": metrics.map_50,
                "map_50_95": metrics.map_50_95,
                "recall_50": metrics.recall_50,
            }
        )
        logger.info(
            "YOLOBenchmark COCO native GT: mAP@[0.5:0.95]=%.3f mAP@0.5=%.3f recall@0.5=%.3f",
            metrics.map_50_95,
            metrics.map_50,
            metrics.recall_50,
        )
        return metrics.map_50_95

    def _evaluate_eval_images(
        self,
        handle: object,
        backend: ComputeBackend,
        manager: ModelManager,
    ) -> float | None:
        """Return mAP@50 against a CPU/ONNX oracle on explicit eval images."""
        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("opencv-python not installed -- skipping YOLO accuracy evaluation")
            return None

        from moment_to_action.hardware import ComputeBackend

        cpu_backend = ComputeBackend(preferred_unit=ComputeUnit.CPU)
        oracle_handle = cpu_backend.load_model(manager.get_path(ModelID.YOLO_V8))

        details = backend.get_input_details(handle)
        shape = details[0]["shape"]
        nhwc = (
            len(shape) == _INPUT_NDIM
            and int(shape[-1]) == _RGB_CHANNELS
            and int(shape[1]) != _RGB_CHANNELS
        )
        if nhwc:
            height_in, width_in = int(shape[1]), int(shape[2])
        else:
            height_in, width_in = int(shape[2]), int(shape[3])

        oracle_preds: list[list[np.ndarray]] = []
        eval_preds: list[list[np.ndarray]] = []
        oracle_size = 640

        for img_path in self._eval_image_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning("Could not load eval image %s -- skipping", img_path)
                continue

            oracle_tensor = _preprocess_nchw(img_bgr, oracle_size, oracle_size)
            oracle_outputs = cpu_backend.run(oracle_handle, oracle_tensor)
            oracle_preds.append(parse_yolo_outputs(oracle_outputs))

            if nhwc:
                eval_tensor = _preprocess_nhwc(img_bgr, height_in, width_in)
            else:
                eval_tensor = _preprocess_nchw(img_bgr, height_in, width_in)
            eval_outputs = backend.run(handle, eval_tensor)

            scale = float(oracle_size) / height_in if height_in != oracle_size else 1.0
            raw_boxes = parse_yolo_outputs(eval_outputs)
            if scale != 1.0:
                raw_boxes = [
                    np.array(
                        [box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale],
                        dtype=np.float32,
                    )
                    for box in raw_boxes
                ]
            eval_preds.append(raw_boxes)

        if not oracle_preds:
            return None

        return compute_map50(eval_preds, oracle_preds)


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


def _default_sample_images() -> list[Path]:
    """Locate the project images/ directory relative to this file."""
    candidate = Path(__file__).parents[4] / "images"
    if candidate.is_dir():
        return sorted(candidate.glob("*.jpg"))
    return []
