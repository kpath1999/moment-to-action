"""YOLOv8 detection model."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np

from moment_to_action.models._formats import ModelFormat
from moment_to_action.models.image.detection._base import ImageDetectionModel
from moment_to_action.models.image.detection._types import BoundingBox, Detection

if TYPE_CHECKING:
    from pathlib import Path

    from moment_to_action.hardware import ComputeBackend

_YOLO_INPUT_SIZE = 640


class YOLOModel(ImageDetectionModel):
    """YOLOv8 object detector.

    Supports ONNX (CPU/GPU via ONNX Runtime) and DLC (NPU via QAIRT) formats.
    The model is unloaded after construction; call :meth:`load` before inference.

    Args:
        variant: Registry variant key used to identify this instance.
        path: Path to the model weights file (``.onnx`` or ``.dlc``).
        model_format: Model file format — determines which backend methods to call.
        confidence_threshold: Minimum confidence score to keep a detection.
    """

    # COCO class labels (80 classes)
    COCO_LABELS: ClassVar[tuple[str, ...]] = (
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    )

    def __init__(
        self,
        variant: str,
        path: Path,
        model_format: ModelFormat,
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initialize an unloaded YOLOModel.

        Args:
            variant: Registry variant key.
            path: Path to the model weights file.
            model_format: ``ModelFormat.ONNX`` or ``ModelFormat.DLC``.
            confidence_threshold: Detections below this score are discarded.
        """
        super().__init__(variant, path)
        self._format = model_format
        self._confidence_threshold = confidence_threshold
        self._handle: object = None

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence score kept by :meth:`post_proc`."""
        return self._confidence_threshold

    def load(self, backend: ComputeBackend) -> None:
        """Load model weights onto the backend.

        Args:
            backend: Hardware backend to load the model onto.

        Raises:
            RuntimeError: If the model is already loaded.
        """
        if self._backend is not None:
            msg = f"{type(self).__name__} is already loaded; call unload() first"
            raise RuntimeError(msg)
        if self._format is ModelFormat.ONNX:
            self._handle = backend.load_model(self._path)
        else:
            self._handle = backend.load_model_dlc(self._path / "model.dlc")
        self._backend = backend

    def unload(self) -> None:
        """Release backend resources and reset internal state."""
        if self._backend is not None:
            if self._format is ModelFormat.ONNX:
                self._backend.unload_model(self._handle)
            else:
                self._backend.unload_dlc(self._handle)
        self._backend = None
        self._handle = None

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, and batch a raw BGR frame for YOLO inference.

        Args:
            frame: Raw BGR image (HxWxC, uint8).

        Returns:
            Float32 tensor of shape ``(1, 3, 640, 640)`` with values in ``[0, 1]``.
        """
        resized = cv2.resize(frame, (_YOLO_INPUT_SIZE, _YOLO_INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        # HxWxC → CxHxW → 1xCxHxW
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def run(self, prepared: np.ndarray) -> list[np.ndarray]:
        """Run YOLOv8 forward pass.

        Args:
            prepared: Batch tensor from :meth:`prepare`.

        Returns:
            List of raw output tensors (boxes, scores, class IDs).

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._backend is None:
            msg = "YOLOModel.load() must be called before run()"
            raise RuntimeError(msg)
        if self._format is ModelFormat.ONNX:
            return self._backend.run(self._handle, prepared)
        return [self._backend.infer_dlc(self._handle, prepared)]

    def post_proc(self, raw: list[np.ndarray]) -> list[Detection]:
        """Decode YOLOv8 3-output format into detections.

        Expects ``raw`` to be a list of three tensors:
        - ``outputs[0]``: ``[1, N, 4]`` float32 — boxes (x1, y1, x2, y2) in 640x640 space
        - ``outputs[1]``: ``[1, N]`` float32 — confidence scores
        - ``outputs[2]``: ``[1, N]`` uint8 — class IDs

        Args:
            raw: Value returned by :meth:`run`.

        Returns:
            Detections above :attr:`confidence_threshold` after NMS, scaled to
            the original frame's pixel coordinates.  Caller must pass the
            original frame size separately (see :meth:`_decode`).
        """
        return self._decode(raw, original_size=None)

    def decode(
        self,
        raw: object,
        original_size: tuple[int, int],
    ) -> list[Detection]:
        """Decode raw output and scale boxes to the original image dimensions.

        Args:
            raw: Value returned by :meth:`run`.
            original_size: ``(height, width)`` of the source frame before preprocessing.

        Returns:
            List of :class:`~moment_to_action.models.image.detection.Detection` objects.
        """
        outputs = list(raw)  # type: ignore[call-overload]
        return self._decode(outputs, original_size=original_size)

    def _decode(
        self,
        outputs: list[np.ndarray],
        original_size: tuple[int, int] | None,
    ) -> list[Detection]:
        """Parse YOLOv8 3-output format, filter, NMS, and scale.

        Args:
            outputs: List of raw output tensors.
            original_size: ``(height, width)`` to scale boxes to; ``None`` keeps 640x640 coords.

        Returns:
            Filtered and NMS-reduced list of detections.
        """
        _expected_output_count = 3
        if len(outputs) < _expected_output_count:
            return []

        boxes_raw = outputs[0][0].astype(np.float32)  # [N, 4]
        scores = outputs[1][0].astype(np.float32)  # [N]
        class_ids = outputs[2][0]  # [N]

        mask = scores >= self._confidence_threshold
        boxes_raw = boxes_raw[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_raw) == 0:
            return []

        keep = self._nms(boxes_raw, scores, iou_threshold=0.45)
        boxes_raw = boxes_raw[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        if original_size is not None:
            orig_h, orig_w = original_size
            sx = orig_w / float(_YOLO_INPUT_SIZE)
            sy = orig_h / float(_YOLO_INPUT_SIZE)
        else:
            sx, sy = 1.0, 1.0
            orig_w, orig_h = _YOLO_INPUT_SIZE, _YOLO_INPUT_SIZE

        detections: list[Detection] = []
        for box, score, cid in zip(boxes_raw, scores, class_ids, strict=False):
            x1 = max(0.0, float(box[0]) * sx)
            y1 = max(0.0, float(box[1]) * sy)
            x2 = min(float(orig_w), float(box[2]) * sx)
            y2 = min(float(orig_h), float(box[3]) * sy)
            class_id = int(cid)
            label = (
                self.COCO_LABELS[class_id] if class_id < len(self.COCO_LABELS) else str(class_id)
            )
            detections.append(
                Detection(
                    label=label,
                    confidence=float(score),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        return detections

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        """Pure NumPy non-maximum suppression.

        Args:
            boxes: ``[N, 4]`` float32 array of (x1, y1, x2, y2) boxes.
            scores: ``[N]`` float32 confidence scores.
            iou_threshold: Boxes with IoU above this threshold are suppressed.

        Returns:
            Indices of boxes to keep, in descending score order.
        """
        indices = np.argsort(scores)[::-1]
        keep: list[int] = []
        while len(indices) > 0:
            cur = indices[0]
            keep.append(int(cur))
            if len(indices) == 1:
                break
            cb = boxes[cur]
            rb = boxes[indices[1:]]
            x1 = np.maximum(cb[0], rb[:, 0])
            y1 = np.maximum(cb[1], rb[:, 1])
            x2 = np.minimum(cb[2], rb[:, 2])
            y2 = np.minimum(cb[3], rb[:, 3])
            inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
            cur_area = (cb[2] - cb[0]) * (cb[3] - cb[1])
            rem_areas = (rb[:, 2] - rb[:, 0]) * (rb[:, 3] - rb[:, 1])
            iou = inter / (cur_area + rem_areas - inter + 1e-6)
            indices = indices[1:][iou < iou_threshold]
        return keep
