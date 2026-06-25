"""YOLOv8 detection model."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import cv2
import numpy as np

from moment_to_action.hardware._types import ModelType
from moment_to_action.models.image.detection._base import ImageDetectionModel
from moment_to_action.models.image.detection._types import BoundingBox, Detection

if TYPE_CHECKING:
    from moment_to_action.hardware import LoadedModel, Platform
    from moment_to_action.hardware._types import ComputeUnit, DataType

_YOLO_OUTPUT_NAME = "output0"
_YOLO_INPUT_SIZE = 640
_YOLO_ANCHORS = 8400


def _strip_qdq(model_proto: object) -> bool:
    """Remove QuantizeLinear/DequantizeLinear pairs from an ONNX graph in-place.

    QAIRT treats any ONNX containing QuantizeLinear nodes as fully pre-quantized
    and skips calibration entirely.  This function removes all Q→DQ pairs so
    QAIRT performs its own INT8 quantization from the supplied calibration data.

    Args:
        model_proto: ``onnx.ModelProto`` to modify in-place.

    Returns:
        ``True`` if any nodes were removed, ``False`` if the graph had no QDQ nodes.
    """
    graph = model_proto.graph  # type: ignore[attr-defined]

    # q_out → float_input for every QuantizeLinear node
    q_output_to_input: dict[str, str] = {}
    for node in graph.node:
        if node.op_type == "QuantizeLinear":
            q_output_to_input[node.output[0]] = node.input[0]

    if not q_output_to_input:
        return False

    # dq_out → original float tensor for every paired DequantizeLinear node
    dq_remap: dict[str, str] = {}
    for node in graph.node:
        if node.op_type == "DequantizeLinear" and node.input[0] in q_output_to_input:
            dq_remap[node.output[0]] = q_output_to_input[node.input[0]]

    # Drop all Q and DQ nodes; remap inputs of surviving nodes
    new_nodes = []
    for node in graph.node:
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        new_inputs = [dq_remap.get(inp, inp) for inp in node.input]
        del node.input[:]
        node.input.extend(new_inputs)
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    # For graph outputs produced by removed DQ nodes, insert an Identity node
    # that maps the original float tensor to the preserved output name.  Renaming
    # out.name would change the DLC tensor key that run() looks up.
    from onnx import helper as oh  # noqa: PLC0415

    for out in graph.output:
        if out.name in dq_remap:
            graph.node.append(oh.make_node("Identity", [dq_remap[out.name]], [out.name]))

    return True


def _expose_cls_for_reducemax(model_proto: object) -> bool:
    """Replace a ReduceMax-based ``scores`` output with the raw transposed class tensor.

    QAIRT CPU has a bug where ``ReduceMax`` on INT8 tensors produces constant
    outputs.  This function removes the ``ReduceMax`` (and any ``Identity`` node
    feeding the ``scores`` graph output) and instead exposes the ``ReduceMax``
    input—the already-transposed ``(1, 8400, 80)`` class tensor—as a ``"cls"``
    graph output.

    :meth:`YOLOModel.run` computes ``scores = dlc_out["cls"].max(axis=-1)``
    in numpy so the reduction happens outside the quantized DLC.

    Args:
        model_proto: ``onnx.ModelProto`` to modify in-place.

    Returns:
        ``True`` if a ``ReduceMax`` was found and replaced, ``False`` otherwise.
    """
    from onnx import TensorProto  # noqa: PLC0415
    from onnx import helper as oh  # noqa: PLC0415

    graph = model_proto.graph  # type: ignore[attr-defined]

    reducemax_node = next((n for n in graph.node if n.op_type == "ReduceMax"), None)
    if reducemax_node is None:
        return False

    cls_input_name: str = reducemax_node.input[0]
    reducemax_output: str = reducemax_node.output[0]

    # Remove ReduceMax and any Identity(reducemax_output → scores) that follows.
    new_nodes = []
    for node in graph.node:
        if node is reducemax_node:
            continue
        if node.op_type == "Identity" and node.input[0] == reducemax_output:
            continue
        new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Expose cls_input_name as a "cls" graph output via Identity (preserves the
    # internal tensor name so QAIRT can still find it as an intermediate output).
    graph.node.append(oh.make_node("Identity", [cls_input_name], ["cls"]))

    # Replace the "scores" entry in graph.output with "cls".
    new_outputs = []
    for out in graph.output:
        if out.name == "scores":
            new_outputs.append(
                oh.make_tensor_value_info("cls", TensorProto.FLOAT, [1, _YOLO_ANCHORS, 80])
            )
        else:
            new_outputs.append(out)
    del graph.output[:]
    graph.output.extend(new_outputs)

    return True


def _split_yolo_concat(model_proto: object) -> bool:
    """Rewrite a single ``output0`` Concat into three homogeneous-range outputs.

    Standard YOLOv8 exports concatenate box coordinates (0-640) and class scores
    (0-1) into a single ``output0 (1, 84, 8400)`` tensor.  QNN assigns one
    quantization scale per output tensor, so the mixed-range tensor causes scores
    to collapse after INT8 quantization.

    This function detects the Concat and replaces it with:
    - ``boxes (1, 8400, 4)`` - decoded bounding boxes
    - ``cls (1, 8400, 80)`` - transposed raw class probabilities (scores computed
      outside the DLC via :meth:`YOLOModel.run`)
    - ``class_idx (1, 8400)`` - per-anchor argmax class index (float32)

    Args:
        model_proto: ``onnx.ModelProto`` to modify in-place.

    Returns:
        ``True`` if the Concat was split, ``False`` if no changes were made.
    """
    from onnx import TensorProto  # noqa: PLC0415
    from onnx import helper as oh  # noqa: PLC0415

    graph = model_proto.graph  # type: ignore[attr-defined]

    out_names = {o.name for o in graph.output}
    if _YOLO_OUTPUT_NAME not in out_names:
        return False

    concat_node = next(
        (n for n in graph.node if _YOLO_OUTPUT_NAME in n.output and n.op_type == "Concat"),
        None,
    )
    if concat_node is None:
        return False

    dbox_name, cls_name = concat_node.input[0], concat_node.input[1]
    boxes_name = "boxes"
    cls_t_name = "_m2a_cls_T"
    cls_out_name = "cls"
    argmax_name = "_m2a_argmax"
    class_idx_name = "class_idx"

    # Transpose cls from (1, 80, 8400) → (1, 8400, 80) so run() can always use
    # axis=-1 for the in-Python ReduceMax regardless of which surgery path ran.
    graph.node.extend(
        [
            oh.make_node("Transpose", [dbox_name], [boxes_name], perm=[0, 2, 1]),
            oh.make_node("Transpose", [cls_name], [cls_t_name], perm=[0, 2, 1]),
            oh.make_node("Identity", [cls_t_name], [cls_out_name]),
            oh.make_node("ArgMax", [cls_name], [argmax_name], axis=1, keepdims=0),
            oh.make_node("Cast", [argmax_name], [class_idx_name], to=TensorProto.FLOAT),
        ]
    )
    graph.node.remove(concat_node)

    _yolo_classes = 80
    del graph.output[:]
    graph.output.extend(
        [
            oh.make_tensor_value_info(boxes_name, TensorProto.FLOAT, [1, _YOLO_ANCHORS, 4]),
            oh.make_tensor_value_info(
                cls_out_name, TensorProto.FLOAT, [1, _YOLO_ANCHORS, _yolo_classes]
            ),
            oh.make_tensor_value_info(class_idx_name, TensorProto.FLOAT, [1, _YOLO_ANCHORS]),
        ]
    )
    return True


class YOLOModel(ImageDetectionModel):
    """YOLOv8 object detector.

    Supports ONNX (CPU/GPU via ONNX Runtime) and DLC (NPU via QAIRT) formats.
    The model is unloaded after construction; call :meth:`load` before inference.

    Args:
        variant: Registry variant key used to identify this instance.
        path: Path to the model weights file (``.onnx`` or ``.dlc``).
        model_type: Model file format — determines which backend methods to call.
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
        model_type: ModelType,
        data_type: DataType,
        confidence_threshold: float = 0.5,
        *,
        backends: dict[ComputeUnit, dict[str, str]],
        input_layout: str = "NCHW",
    ) -> None:
        """Initialize an unloaded YOLOModel.

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
        """
        super().__init__(
            variant, path, model_type, data_type, backends=backends, input_layout=input_layout
        )
        self._confidence_threshold = confidence_threshold
        self._handle: LoadedModel | None = None
        self._last_original_size: tuple[int, int] | None = None

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence score kept by :meth:`post_proc`."""
        return self._confidence_threshold

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

    def prepare_for_conversion(self, onnx_path: Path) -> Path:
        """Apply ONNX surgery to prepare for DLC conversion.

        Performs two transformations as needed:

        1. **QDQ stripping**: removes QuantizeLinear/DequantizeLinear pairs so
           QAIRT performs its own INT8 quantization from calibration data rather
           than reusing embedded scales (which produce collapsed scores on HTP).

        2. **Concat split**: if the model has a single ``output0 (1, 84, 8400)``
           tensor that mixes box coordinates (0-640) and class scores (0-1),
           rewrites it into three homogeneous-range outputs so each gets its own
           INT8 scale:
           - ``boxes (1, 8400, 4)`` - decoded bounding boxes
           - ``cls (1, 8400, 80)`` - transposed raw class probabilities
           - ``class_idx (1, 8400)`` - per-anchor argmax class index

        3. **ReduceMax replacement**: replaces any ``ReduceMax``-based ``scores``
           output with the raw transposed class tensor ``cls (1, 8400, 80)``.
           QAIRT CPU has a bug producing constant outputs from quantized ReduceMax;
           :meth:`run` computes ``scores`` via ``dlc_out["cls"].max(axis=-1)``
           in numpy instead.

        Args:
            onnx_path: Path to the source YOLOv8 ONNX.

        Returns:
            ``onnx_path`` if no surgery was needed, otherwise a path to a
            temporary file that the caller must delete after conversion.
        """
        import onnx  # noqa: PLC0415

        model_proto = onnx.load(str(onnx_path))

        changed = _strip_qdq(model_proto)
        changed |= _split_yolo_concat(model_proto)
        changed |= _expose_cls_for_reducemax(model_proto)

        if not changed:
            return onnx_path

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        onnx.save(model_proto, str(tmp_path))
        return tmp_path

    @property
    def input_layout(self) -> str:
        """Input tensor layout: ``"NCHW"`` or ``"NHWC"`` (set at construction from the Variant)."""
        return self._input_layout or "NCHW"

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, and batch a raw BGR frame for YOLO inference.

        Args:
            frame: Raw BGR image (HxWxC, uint8).

        Returns:
            Float32 tensor with values in ``[0, 1]``:

            - ``(1, 3, 640, 640)`` when :attr:`input_layout` is ``"NCHW"``
              (ONNX / local-convert DLC).
            - ``(1, 640, 640, 3)`` when :attr:`input_layout` is ``"NHWC"``
              (AI Hub DLC — the model was exported with NHWC input).
        """
        self._last_original_size = (frame.shape[0], frame.shape[1])
        resized = cv2.resize(frame, (_YOLO_INPUT_SIZE, _YOLO_INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        if self._input_layout == "NHWC":
            # Keep HxWxC, just add batch dim → (1, 640, 640, 3)
            return np.expand_dims(normalized, axis=0)
        # HxWxC → CxHxW → 1xCxHxW (NCHW)
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def _run(self, prepared: np.ndarray) -> list[np.ndarray]:
        """Run YOLOv8 forward pass.

        Args:
            prepared: Batch tensor from :meth:`prepare`.

        Returns:
            List of raw output tensors (boxes, scores, class IDs).

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._handle is None:
            msg = "YOLOModel.load() must be called before run()"
            raise RuntimeError(msg)
        if self._model_type is ModelType.ONNX:
            return cast("list[np.ndarray]", self._handle.run(prepared))
        dlc_out = cast("dict[str, np.ndarray]", self._handle.run(prepared))
        if "scores" in dlc_out:
            # AI Hub DLC already emits per-anchor max class prob directly.
            return [dlc_out["boxes"], dlc_out["scores"], dlc_out["class_idx"]]
        # Local-convert DLC outputs "cls" (1, 8400, 80) to avoid QAIRT CPU's
        # broken ReduceMax on INT8.  Compute max class prob in numpy.
        scores = dlc_out["cls"].max(axis=dlc_out["cls"].ndim - 1)
        return [dlc_out["boxes"], scores, dlc_out["class_idx"]]

    def _post_proc(self, raw: list[np.ndarray]) -> list[Detection]:
        """Decode YOLOv8 3-output format into detections.

        Expects ``raw`` to be a list of three tensors:
        - ``outputs[0]``: ``[1, N, 4]`` float32 — boxes (x1, y1, x2, y2) in 640x640 space
        - ``outputs[1]``: ``[1, N]`` float32 — confidence scores
        - ``outputs[2]``: ``[1, N]`` uint8 — class IDs

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
