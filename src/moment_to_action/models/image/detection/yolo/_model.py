"""YOLOv8 detection model."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np

from moment_to_action.models._formats import ModelFormat
from moment_to_action.models.image.detection._base import ImageDetectionModel
from moment_to_action.models.image.detection._types import BoundingBox, Detection

if TYPE_CHECKING:
    from moment_to_action.hardware import ComputeBackend

_YOLO_OUTPUT_NAME = "output0"
_YOLO_INPUT_SIZE = 640
_YOLO_ANCHORS = 8400
_ONNX_OPSET_AXES_AS_INPUT = 18


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

    def prepare_for_conversion(self, onnx_path: Path) -> Path:
        """Split YOLOv8 output0 Concat before DLC conversion if needed.

        Standard YOLOv8 exports produce a single ``output0 (1, 84, 8400)`` that
        concatenates box coordinates (0-640) and class scores (0-1).  QNN assigns
        one quantization scale per output tensor, so a mixed-range tensor causes
        the 0-1 scores to collapse to zero after INT8 quantization.

        This method detects that Concat and rewrites the graph to expose three
        homogeneous-range outputs instead:
        - ``boxes (1, 8400, 4)`` - decoded bounding boxes
        - ``scores (1, 8400)`` - per-anchor max class probability
        - ``class_idx (1, 8400)`` - per-anchor argmax class index (float32)

        If the ONNX already has split outputs (e.g. the vendored model), returns
        ``onnx_path`` unchanged.

        Args:
            onnx_path: Path to the source YOLOv8 ONNX.

        Returns:
            ``onnx_path`` if no surgery was needed, otherwise a path to a
            temporary file that the caller must delete after conversion.
        """
        import onnx  # noqa: PLC0415
        from onnx import TensorProto  # noqa: PLC0415
        from onnx import helper as oh  # noqa: PLC0415

        model_proto = onnx.load(str(onnx_path))
        graph = model_proto.graph

        out_names = {o.name for o in graph.output}
        if _YOLO_OUTPUT_NAME not in out_names:
            return onnx_path  # already split

        concat_node = next(
            (n for n in graph.node if _YOLO_OUTPUT_NAME in n.output and n.op_type == "Concat"),
            None,
        )
        if concat_node is None:
            return onnx_path

        dbox_name, cls_name = concat_node.input[0], concat_node.input[1]
        boxes_name = "_m2a_boxes"
        scores_name = "_m2a_scores"
        argmax_name = "_m2a_argmax"
        class_idx_name = "_m2a_class_idx"

        # ReduceMax: axes is an attribute (opset < 18) or an input tensor (opset >= 18)
        opset = next(
            (op.version for op in model_proto.opset_import if op.domain in {"", "ai.onnx"}),
            11,
        )
        if opset >= _ONNX_OPSET_AXES_AS_INPUT:
            axes_init_name = "_m2a_reduce_axes"
            graph.initializer.append(oh.make_tensor(axes_init_name, TensorProto.INT64, [1], [1]))
            reduce_node = oh.make_node(
                "ReduceMax", [cls_name, axes_init_name], [scores_name], keepdims=0
            )
        else:
            reduce_node = oh.make_node("ReduceMax", [cls_name], [scores_name], axes=[1], keepdims=0)

        graph.node.extend(
            [
                oh.make_node("Transpose", [dbox_name], [boxes_name], perm=[0, 2, 1]),
                reduce_node,
                oh.make_node("ArgMax", [cls_name], [argmax_name], axis=1, keepdims=0),
                oh.make_node("Cast", [argmax_name], [class_idx_name], to=TensorProto.FLOAT),
            ]
        )
        graph.node.remove(concat_node)

        del graph.output[:]
        graph.output.extend(
            [
                oh.make_tensor_value_info(boxes_name, TensorProto.FLOAT, [1, _YOLO_ANCHORS, 4]),
                oh.make_tensor_value_info(scores_name, TensorProto.FLOAT, [1, _YOLO_ANCHORS]),
                oh.make_tensor_value_info(class_idx_name, TensorProto.FLOAT, [1, _YOLO_ANCHORS]),
            ]
        )

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        onnx.save(model_proto, str(tmp_path))
        return tmp_path

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, and batch a raw BGR frame for YOLO inference.

        ONNX models expect NCHW ``(1, 3, 640, 640)``.
        DLC models expect NHWC ``(1, 640, 640, 3)`` because qnn-onnx-converter
        transposes NCHW → NHWC internally when targeting HTP.

        Args:
            frame: Raw BGR image (HxWxC, uint8).

        Returns:
            Float32 tensor with values in ``[0, 1]``.
            Shape is ``(1, 3, 640, 640)`` for ONNX and ``(1, 640, 640, 3)`` for DLC.
        """
        resized = cv2.resize(frame, (_YOLO_INPUT_SIZE, _YOLO_INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        if self._format is ModelFormat.DLC:
            # HxWxC → 1xHxWxC (NHWC)
            return np.expand_dims(normalized, axis=0)
        # HxWxC → CxHxW → 1xCxHxW (NCHW)
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
        dlc_out = self._backend.infer_dlc(self._handle, prepared)
        return [dlc_out["boxes"], dlc_out["scores"], dlc_out["class_idx"]]

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
