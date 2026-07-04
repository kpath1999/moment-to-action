"""RF-DETR object detection model."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

import cv2
import numpy as np

from moment_to_action.hardware._types import ModelType
from moment_to_action.models.image.detection._base import ImageDetectionModel
from moment_to_action.models.image.detection._types import BoundingBox, Detection

if TYPE_CHECKING:
    from pathlib import Path

    from moment_to_action.hardware import LoadedModel, Platform
    from moment_to_action.hardware._types import ComputeUnit, DataType
    from moment_to_action.metrics import MetricsCollector

_RF_DETR_INPUT_SIZE = 560


class RFDETRModel(ImageDetectionModel):
    """RF-DETR object detector.

    Supports ONNX (CPU/GPU via ONNX Runtime) and DLC (NPU via QAIRT) formats.
    The model is unloaded after construction; call :meth:`load` before inference.

    RF-DETR outputs three homogeneous-range tensors after AI Hub export:
    ``boxes [1, N, 4]``, ``scores [1, N]``, ``class_idx [1, N]``.

    Args:
        variant: Registry variant key used to identify this instance.
        path: Path to the model weights file (``.onnx`` or ``.dlc``).
        model_type: Model file format — determines which backend methods to call.
        confidence_threshold: Minimum confidence score to keep a detection.
    """

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
        model_type: ModelType,
        data_type: DataType,
        confidence_threshold: float = 0.5,
        *,
        backends: dict[ComputeUnit, dict[str, str]],
        input_layout: str = "NCHW",
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialize an unloaded RFDETRModel.

        Args:
            variant: Registry variant key.
            path: Path to the model weights file or variant directory.
            model_type: ``ModelType.ONNX`` or ``ModelType.DLC``.
            data_type: Quantization type (e.g. ``DataType.W8A8``); required for DLC variants.
            confidence_threshold: Detections below this score are discarded.
            backends: Compute unit → ``{"model": filename}`` mapping.  Keys
                present are the supported units; ``load()`` indexes this with
                the explicit ``unit`` argument.
            input_layout: ``"NCHW"`` or ``"NHWC"``.  QCS6490 AI Hub DLC exports
                use ``"NHWC"``; all other variants use ``"NCHW"`` (default).
            metrics: Metrics collector used to record ``MODEL_*`` spans.
        """
        super().__init__(
            variant,
            path,
            model_type,
            data_type,
            backends=backends,
            input_layout=input_layout,
            metrics=metrics,
        )
        self._confidence_threshold = confidence_threshold
        self._handle: LoadedModel | None = None
        self._last_original_size: tuple[int, int] | None = None

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence score kept by :meth:`post_proc`."""
        return self._confidence_threshold

    @property
    def input_layout(self) -> str:
        """Input tensor layout: ``"NCHW"`` or ``"NHWC"`` (set at construction from the Variant)."""
        return self._input_layout or "NCHW"

    def _load(self, platform: Platform, unit: ComputeUnit) -> None:
        """Load model weights onto the backend.

        Selects the artifact filename from the per-unit ``backends`` table
        using ``unit``.

        Args:
            platform: Hardware platform to load the model onto.
            unit: Compute unit to target.

        Raises:
            RuntimeError: If the model is already loaded.
            KeyError: If ``unit`` is not supported by this variant.
            ValueError: If ``unit`` is not available on ``platform``.
        """
        if self._platform is not None:
            msg = f"{type(self).__name__} is already loaded; call unload() first"
            raise RuntimeError(msg)
        arts = self._backends[unit]
        if self._model_type is ModelType.ONNX:
            dtype = self._data_type
            self._handle = platform.load_onnx(unit, self._artifact_path(arts["model"]), dtype=dtype)
        else:
            dtype = self._data_type
            self._handle = platform.load_dlc(unit, self._artifact_path(arts["model"]), dtype=dtype)
        self._platform = platform

    def _unload(self) -> None:
        """Release backend resources and reset internal state."""
        if self._handle is not None:
            self._handle.unload()
        self._platform = None
        self._handle = None

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, and batch a raw BGR frame for RF-DETR inference.

        Args:
            frame: Raw BGR image (HxWxC, uint8).

        Returns:
            Float32 tensor with values in ``[0, 1]``:

            - ``(1, 3, 560, 560)`` when :attr:`input_layout` is ``"NCHW"``.
            - ``(1, 560, 560, 3)`` when :attr:`input_layout` is ``"NHWC"``.
        """
        self._last_original_size = (frame.shape[0], frame.shape[1])
        resized = cv2.resize(frame, (_RF_DETR_INPUT_SIZE, _RF_DETR_INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        if self._input_layout == "NHWC":
            return np.expand_dims(normalized, axis=0)
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def _run(self, prepared: np.ndarray) -> list[np.ndarray]:
        """Run RF-DETR forward pass.

        Args:
            prepared: Batch tensor from :meth:`prepare`.

        Returns:
            List of raw output tensors ``[boxes, scores, class_idx]``.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._handle is None:
            msg = "RFDETRModel.load() must be called before run()"
            raise RuntimeError(msg)
        if self._model_type is ModelType.ONNX:
            return cast("list[np.ndarray]", self._handle.run(prepared))
        dlc_out = cast("dict[str, np.ndarray]", self._handle.run(prepared))
        # AI Hub export uses "logits" (post-softmax confidence) and "classes" instead of
        # "scores" / "class_idx" — remap to the uniform [boxes, scores, class_idx] contract.
        return [dlc_out["boxes"], dlc_out["logits"], dlc_out["classes"]]

    def _post_proc(self, raw: list[np.ndarray]) -> list[Detection]:
        """Decode RF-DETR 3-output format into detections.

        Args:
            raw: Value returned by :meth:`run`.

        Returns:
            Detections above :attr:`confidence_threshold` after NMS, scaled to
            the original frame's pixel coordinates using the size recorded by
            the preceding :meth:`prepare` call.
        """
        return self._decode(raw, original_size=self._last_original_size)

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
        return self._decode(list(raw), original_size=original_size)  # type: ignore[call-overload]

    def _decode(
        self,
        outputs: list[np.ndarray],
        original_size: tuple[int, int] | None,
    ) -> list[Detection]:
        """Parse RF-DETR 3-output format, filter, NMS, and scale.

        Args:
            outputs: List of raw output tensors ``[boxes, scores, class_idx]``.
            original_size: ``(height, width)`` to scale boxes to; ``None`` keeps model-space coords.

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
            sx = orig_w / float(_RF_DETR_INPUT_SIZE)
            sy = orig_h / float(_RF_DETR_INPUT_SIZE)
        else:
            sx, sy = 1.0, 1.0
            orig_w, orig_h = _RF_DETR_INPUT_SIZE, _RF_DETR_INPUT_SIZE

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
