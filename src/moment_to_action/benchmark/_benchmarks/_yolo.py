from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from moment_to_action.benchmark._benchmarks._base import ModelBenchmark
from moment_to_action.benchmark._detection_metrics import compute_detection_map
from moment_to_action.benchmark._oracle_ground_truth import OracleBox, OracleDetection
from moment_to_action.models import ModelID
from moment_to_action.stages.video._yolo import YOLOStage

if TYPE_CHECKING:
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
_BOX_INSPECTION_IMAGES = 5
_MAX_DEBUG_LIST_ITEMS = 10
_TOPK_LABELS = 5
_DEBUG_OUTPUT_DIR = Path("logs/yolo_debug")
_PROB_MIN = -1e-4
_PROB_MAX = 1.0001
_EMPTY_MASK_DEBUG_TOPK = 10
_GT_COLOR = (0, 200, 0)
_PRED_COLOR = (220, 20, 60)
_HIGH_PASS_RATIO = 0.95
# Default confidence threshold for GPU TFLite path: FP16 models produce sparser
# class probability distributions than CPU/NPU paths, so a lower threshold is
# needed to recover detections.
_GPU_CONF_THRESHOLD = 0.05


def _box_iou(box_a: OracleBox, box_b: OracleBox) -> float:
    """Compute IoU between two OracleBox instances in pixel coordinates."""
    inter_x1 = max(box_a.x1, box_b.x1)
    inter_y1 = max(box_a.y1, box_b.y1)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, box_a.x2 - box_a.x1) * max(0.0, box_a.y2 - box_a.y1)
    area_b = max(0.0, box_b.x2 - box_b.x1) * max(0.0, box_b.y2 - box_b.y1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def _label_key(label: str) -> str:
    """Normalize labels for easier GT-vs-pred mismatch inspection."""
    return " ".join(label.strip().lower().split())


def _looks_like_probability(arr: np.ndarray) -> bool:
    """Return True when values are already in [0, 1] (allowing small numeric slack)."""
    return float(arr.min()) >= _PROB_MIN and float(arr.max()) <= _PROB_MAX


def _activate_maybe_logits(arr: np.ndarray) -> np.ndarray:
    """Return probabilities, applying sigmoid only when channels look logit-like."""
    if _looks_like_probability(arr):
        return np.clip(arr, 0.0, 1.0)
    return 1 / (1 + np.exp(-arr))


def _effective_conf_threshold(
    base_threshold: float,
    per_unit_thresholds: dict[str, float] | None,
    backend: object,
) -> float:
    """Return the conf threshold to use, respecting per-unit overrides.

    The GPU TFLite FP16 path typically produces much sparser/lower class
    probability distributions than CPU or NPU paths for the same model file.
    Callers pass a *per_unit_thresholds* dict keyed by ``ComputeUnit.value``
    strings (e.g. ``{"gpu": 0.05}``) to lower the threshold for those units.
    """
    if per_unit_thresholds and hasattr(backend, "active_unit"):
        unit_key = getattr(backend.active_unit, "value", None)
        if unit_key is not None:
            # Keys are compared case-insensitively so callers may use either
            # "gpu" or "GPU" without ambiguity.
            override = per_unit_thresholds.get(unit_key) or per_unit_thresholds.get(
                unit_key.lower()
            )
            if override is not None:
                logger.info(
                    "[YOLO PARSE] Using per-unit conf_threshold=%.6f for unit=%s (base=%.6f)",
                    override,
                    unit_key,
                    base_threshold,
                )
                return override
    return base_threshold


def _count_pre_nms_candidates(raw_outputs: object) -> int | None:
    """Return number of raw candidate boxes before NMS, or None if not determinable."""
    if not (
        isinstance(raw_outputs, (list, tuple))
        and len(raw_outputs) > 0
        and isinstance(raw_outputs[0], np.ndarray)
    ):
        return None
    arr: np.ndarray = raw_outputs[0]
    if arr.ndim == _NMS_DIM_3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != _NMS_DIM_2:
        return None
    # Combined output can be [84, N] or [N, 84]. Count the candidate axis.
    if _YOLO_FEATURE_DIM in arr.shape and arr.shape[0] != arr.shape[1]:
        return arr.shape[1] if arr.shape[0] == _YOLO_FEATURE_DIM else arr.shape[0]
    return int(arr.shape[0])


def _letterbox_params(
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
) -> tuple[float, float, float]:
    """Return resize scale and padding used by letterbox preprocessing."""
    scale = min(target_w / orig_w, target_h / orig_h)
    resized_w = orig_w * scale
    resized_h = orig_h * scale
    pad_left = (target_w - resized_w) / 2.0
    pad_top = (target_h - resized_h) / 2.0
    return scale, pad_left, pad_top


def _project_tflite_box_to_original(
    box: OracleBox,
    *,
    orig_w: int,
    orig_h: int,
    model_w: int,
    model_h: int,
) -> OracleBox:
    """Map a TFLite YOLO box from letterboxed model space back to original pixels."""

    def clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    scale, pad_left, pad_top = _letterbox_params(orig_w, orig_h, model_w, model_h)
    is_normalized = all(0.0 <= coord <= 1.0 for coord in (box.x1, box.y1, box.x2, box.y2))

    x1_model = box.x1 * model_w if is_normalized else box.x1
    y1_model = box.y1 * model_h if is_normalized else box.y1
    x2_model = box.x2 * model_w if is_normalized else box.x2
    y2_model = box.y2 * model_h if is_normalized else box.y2

    x1 = clamp((x1_model - pad_left) / scale, 0.0, float(orig_w))
    y1 = clamp((y1_model - pad_top) / scale, 0.0, float(orig_h))
    x2 = clamp((x2_model - pad_left) / scale, 0.0, float(orig_w))
    y2 = clamp((y2_model - pad_top) / scale, 0.0, float(orig_h))

    return OracleBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        label=box.label,
        confidence=box.confidence,
    )


def _find_best_iou_match(gt: OracleBox, pred_boxes: list[OracleBox]) -> tuple[int, float]:
    """Return (index, iou) of the pred box with the highest IoU against *gt*."""
    best_idx = -1
    best_iou = -1.0
    for i, pred in enumerate(pred_boxes):
        cur_iou = _box_iou(gt, pred)
        if cur_iou > best_iou:
            best_iou = cur_iou
            best_idx = i
    return best_idx, best_iou


class YOLOBenchmark(ModelBenchmark):
    def _predict_image(
        self,
        img_path: Path,
        gt_det: OracleDetection,
        handle: object,
        backend: ComputeBackend,
    ) -> OracleDetection:
        """Run prediction and post-processing for a single image."""
        debug_idx = self._debug_image_counter
        self._debug_image_counter += 1

        img_tensor = _load_yolo_tensor(
            img_path,
            self._input_shape,
            getattr(self, "_input_quant", None),
        )
        if debug_idx < _GPU_DEBUG_LOG_IMAGES:
            self._log_tensor_stats("input", img_tensor)

        raw_outputs = backend.run(handle, img_tensor)
        if debug_idx < _GPU_DEBUG_LOG_IMAGES:
            self._log_backend_output_shapes(raw_outputs)

        # Dequantize outputs if needed (NPU quantized model)
        output_quant = getattr(self, "_output_quant", None)
        if output_quant is not None and isinstance(raw_outputs, (list, tuple)):
            from typing import Any

            deq_outputs: list[Any] = []
            for i, out in enumerate(raw_outputs):
                if isinstance(out, np.ndarray) and out.dtype == output_quant["dtype"]:
                    deq = (out.astype(np.float32) - output_quant["zero_point"]) * output_quant[
                        "scale"
                    ]
                    logger.info(
                        "[YOLO NPU] Dequantized output[%d]: min=%.6f, max=%.6f, shape=%s",
                        i,
                        deq.min(),
                        deq.max(),
                        deq.shape,
                    )
                    deq_outputs.append(deq)
                else:
                    deq_outputs.append(out)
            raw_outputs = tuple(deq_outputs)  # type: ignore[assignment]
        elif (
            output_quant is not None
            and isinstance(raw_outputs, np.ndarray)
            and raw_outputs.dtype == output_quant["dtype"]
        ):
            raw_outputs = (
                raw_outputs.astype(np.float32) - output_quant["zero_point"]
            ) * output_quant["scale"]
            logger.info(
                "[YOLO NPU] Dequantized output: min=%.6f, max=%.6f, shape=%s",
                raw_outputs.min(),
                raw_outputs.max(),
                raw_outputs.shape,
            )
        # Pre-thresholding: count number of raw candidate boxes (if possible)
        pre_nms_count = _count_pre_nms_candidates(raw_outputs)
        effective_threshold = _effective_conf_threshold(
            self._conf_threshold,
            self._per_unit_conf_thresholds,
            backend,
        )
        yolo_boxes = _parse_yolo_boxes(
            raw_outputs,
            self._input_shape,
            conf_threshold=effective_threshold,
            class_labels=YOLOStage.COCO_LABELS,
        )
        post_nms_count = len(yolo_boxes)

        self._log_gpu_debug(
            debug_idx,
            raw_outputs,
            gt_det,
            yolo_boxes,
            pre_nms_count,
            post_nms_count,
            img_path=img_path,
        )
        scaled_boxes = self._scale_predicted_boxes(yolo_boxes, img_path)
        self._log_gt_pred_alignment(
            debug_idx=debug_idx,
            image_name=gt_det.image_name,
            gt_boxes=gt_det.boxes,
            pred_boxes=scaled_boxes,
            img_path=img_path,
        )
        return OracleDetection(image_name=gt_det.image_name, boxes=scaled_boxes)

    def _log_tensor_stats(self, name: str, arr: np.ndarray) -> None:
        """Log compact distribution stats to detect quantization/layout issues quickly."""
        flat = arr.astype(np.float32).reshape(-1)
        logger.info(
            "[YOLO DEBUG] Tensor %s: shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f "
            "std=%.6f p01=%.6f p99=%.6f",
            name,
            arr.shape,
            arr.dtype,
            float(flat.min()),
            float(flat.max()),
            float(flat.mean()),
            float(flat.std()),
            float(np.percentile(flat, 1)),
            float(np.percentile(flat, 99)),
        )

    def _log_backend_output_shapes(self, raw_outputs: object) -> None:
        """Log every output tensor shape/dtype to inspect backend-specific output contracts."""
        if isinstance(raw_outputs, dict):
            logger.info(
                "[YOLO DEBUG] Backend output container: dict with %d item(s)",
                len(raw_outputs),
            )
            for name, out in raw_outputs.items():
                if isinstance(out, np.ndarray):
                    self._log_tensor_stats(f"output[{name}]", out)
            return

        if isinstance(raw_outputs, (list, tuple)):
            logger.info(
                "[YOLO DEBUG] Backend output container: %s with %d tensor(s)",
                type(raw_outputs).__name__,
                len(raw_outputs),
            )
            for i, out in enumerate(raw_outputs):
                if isinstance(out, np.ndarray):
                    self._log_tensor_stats(f"output[{i}]", out)
            return

        if isinstance(raw_outputs, np.ndarray):
            logger.info("[YOLO DEBUG] Backend output container: ndarray")
            self._log_tensor_stats("output", raw_outputs)
            return

        logger.warning("[YOLO DEBUG] Unexpected backend output type: %s", type(raw_outputs))

    def _get_output_quant(self, model_path: str) -> dict | None:
        """Read output quantization info from TFLite model if quantized."""
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite
        interpreter = tflite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        output_details = interpreter.get_output_details()[0]
        scale = output_details.get("quantization", (1.0, 0))[0]
        zero_point = output_details.get("quantization", (1.0, 0))[1]
        dtype = output_details.get("dtype", None)
        logger.info(
            "[YOLO NPU] Output quantization: scale=%.6f, zero_point=%d, dtype=%s",
            scale,
            zero_point,
            dtype,
        )
        if dtype is not None and dtype is not float:
            return {"scale": scale, "zero_point": zero_point, "dtype": dtype}
        return None

    def _log_gpu_debug(
        self,
        idx: int,
        raw_outputs: object,
        gt_det: OracleDetection,
        yolo_boxes: list[OracleBox],
        pre_nms_count: int | None,
        post_nms_count: int,
        img_path: Path | None = None,
    ) -> None:
        if idx >= _GPU_DEBUG_LOG_IMAGES:
            return
        self._log_gpu_raw_output(raw_outputs)
        self._log_gpu_box_counts(gt_det, pre_nms_count, post_nms_count)
        self._log_gpu_pred_boxes(yolo_boxes, img_path)
        self._log_gpu_gt_boxes(gt_det)

    def _log_gpu_raw_output(self, raw_outputs: object) -> None:
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

    def _log_gpu_box_counts(
        self,
        gt_det: OracleDetection,
        pre_nms_count: int | None,
        post_nms_count: int,
    ) -> None:
        logger.info("[GPU DEBUG] Image: %s", gt_det.image_name)
        logger.info("[GPU DEBUG] GT boxes: %d", len(gt_det.boxes))
        logger.info("[GPU DEBUG] Pre-NMS candidate boxes: %s", pre_nms_count)
        logger.info("[GPU DEBUG] Post-NMS boxes: %s", post_nms_count)

    def _log_gpu_pred_boxes(
        self,
        yolo_boxes: list[OracleBox],
        img_path: Path | None = None,
    ) -> None:
        if yolo_boxes:
            for i, b in enumerate(yolo_boxes[:_GPU_DEBUG_LOG_IMAGES]):
                logger.info(
                    "[GPU DEBUG] Pred box (normalized) %d: x1=%.3f, y1=%.3f, x2=%.3f, "
                    "y2=%.3f, label=%s, conf=%.3f",
                    i,
                    b.x1,
                    b.y1,
                    b.x2,
                    b.y2,
                    b.label,
                    b.confidence,
                )
            if img_path is not None:
                try:
                    with Image.open(img_path) as _pil:
                        orig_w, orig_h = _pil.size
                    for i, b in enumerate(yolo_boxes[:_GPU_DEBUG_LOG_IMAGES]):
                        logger.info(
                            "[GPU DEBUG] Pred box (pixels) %d: x1=%.1f, y1=%.1f, x2=%.1f, "
                            "y2=%.1f, label=%s, conf=%.3f",
                            i,
                            b.x1 * orig_w,
                            b.y1 * orig_h,
                            b.x2 * orig_w,
                            b.y2 * orig_h,
                            b.label,
                            b.confidence,
                        )
                except (OSError, ValueError) as e:
                    logger.warning("[GPU DEBUG] Could not log pixel boxes: %r", e)
        else:
            logger.info("[GPU DEBUG] No predicted boxes after NMS/threshold.")

    def _log_gpu_gt_boxes(self, gt_det: OracleDetection) -> None:
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

    def _log_gt_pred_alignment(
        self,
        *,
        debug_idx: int,
        image_name: str,
        gt_boxes: list[OracleBox],
        pred_boxes: list[OracleBox],
        img_path: Path,
    ) -> None:
        """Log IoU/label agreement diagnostics and optionally write an overlay image."""
        if debug_idx >= _BOX_INSPECTION_IMAGES:
            return

        logger.info(
            "[YOLO ALIGN] image=%s gt_count=%d pred_count=%d",
            image_name,
            len(gt_boxes),
            len(pred_boxes),
        )

        gt_label_counts = Counter(_label_key(b.label) for b in gt_boxes)
        pred_label_counts = Counter(_label_key(b.label) for b in pred_boxes)
        logger.info(
            "[YOLO ALIGN] GT label histogram (top %d): %s",
            _TOPK_LABELS,
            gt_label_counts.most_common(_TOPK_LABELS),
        )
        logger.info(
            "[YOLO ALIGN] Pred label histogram (top %d): %s",
            _TOPK_LABELS,
            pred_label_counts.most_common(_TOPK_LABELS),
        )

        if not gt_boxes or not pred_boxes:
            logger.info("[YOLO ALIGN] Skipping IoU match stats (one side has no boxes).")
            self._write_debug_overlay(img_path, image_name, gt_boxes, pred_boxes)
            return

        label_match_count = 0
        iou_match_count = 0
        joint_match_count = 0
        logged_pairs = 0

        for gt in gt_boxes:
            best_idx, best_iou = _find_best_iou_match(gt, pred_boxes)

            if best_idx < 0:
                continue

            pred = pred_boxes[best_idx]
            label_match = _label_key(pred.label) == _label_key(gt.label)
            if label_match:
                label_match_count += 1
            if best_iou >= _IOU_RECALL_THRESHOLD:
                iou_match_count += 1
            if label_match and best_iou >= _IOU_RECALL_THRESHOLD:
                joint_match_count += 1

            if logged_pairs < _MAX_DEBUG_LIST_ITEMS:
                logger.info(
                    "[YOLO ALIGN] GT[%d] label=%s best_pred[%d] label=%s iou=%.3f "
                    "pred_conf=%.3f label_match=%s",
                    logged_pairs,
                    gt.label,
                    best_idx,
                    pred.label,
                    best_iou,
                    pred.confidence,
                    label_match,
                )
                logged_pairs += 1

        logger.info(
            "[YOLO ALIGN] match summary: gt=%d label_match=%d iou@%.2f=%d joint(label+iou)=%d",
            len(gt_boxes),
            label_match_count,
            _IOU_RECALL_THRESHOLD,
            iou_match_count,
            joint_match_count,
        )

        self._write_debug_overlay(img_path, image_name, gt_boxes, pred_boxes)

    def _write_debug_overlay(
        self,
        img_path: Path,
        image_name: str,
        gt_boxes: list[OracleBox],
        pred_boxes: list[OracleBox],
    ) -> None:
        """Write a side-by-side style overlay image with GT and predicted boxes."""
        _DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = _DEBUG_OUTPUT_DIR / f"{Path(image_name).stem}_gt_pred_overlay.png"

        try:
            with Image.open(img_path).convert("RGB") as src_img:
                canvas = src_img.copy()

            draw = ImageDraw.Draw(canvas)
            for gt in gt_boxes[:_MAX_DEBUG_LIST_ITEMS]:
                draw.rectangle([gt.x1, gt.y1, gt.x2, gt.y2], outline=_GT_COLOR, width=2)
                draw.text((gt.x1, max(0.0, gt.y1 - 12)), f"GT:{gt.label}", fill=_GT_COLOR)

            for pred in pred_boxes[:_MAX_DEBUG_LIST_ITEMS]:
                draw.rectangle([pred.x1, pred.y1, pred.x2, pred.y2], outline=_PRED_COLOR, width=2)
                draw.text(
                    (pred.x1, pred.y1 + 2),
                    f"PR:{pred.label} {pred.confidence:.2f}",
                    fill=_PRED_COLOR,
                )

            draw.text((8, 8), "Green=GT, Red=Pred", fill=(255, 255, 255))
            canvas.save(output_path)
            logger.info("[YOLO ALIGN] Wrote overlay: %s", output_path)
        except OSError as exc:
            logger.warning("[YOLO ALIGN] Failed to write overlay for %s: %r", image_name, exc)

    def _scale_predicted_boxes(
        self,
        yolo_boxes: list[OracleBox],
        img_path: Path,
    ) -> list[OracleBox]:
        if not yolo_boxes:
            return []
        with Image.open(img_path) as _pil:
            orig_w, orig_h = _pil.size

        if self._is_tflite:
            _, model_h, model_w, _ = self._input_shape
            return [
                _project_tflite_box_to_original(
                    b,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    model_w=model_w,
                    model_h=model_h,
                )
                for b in yolo_boxes
            ]

        # Fallback: original logic for ONNX or other models
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
        per_unit_conf_thresholds: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self._coco_dataset = coco_dataset
        self._conf_threshold = conf_threshold
        self._model_path = model_path
        self._is_tflite = model_path is not None and str(model_path).endswith(".tflite")
        # Default input shape; will be set in _load_model
        self._input_shape: tuple[int, ...] = (1, 3, 640, 640)
        self._debug_image_counter = 0
        # Per-unit confidence threshold overrides.  If None, defaults to
        # {"gpu": _GPU_CONF_THRESHOLD} so the GPU TFLite FP16 path is usable
        # out of the box without manual tuning.
        self._per_unit_conf_thresholds: dict[str, float] = (
            per_unit_conf_thresholds
            if per_unit_conf_thresholds is not None
            else {"gpu": _GPU_CONF_THRESHOLD}
        )

    @property
    def model_id(self) -> ModelID:
        return ModelID.YOLO_V12_N

    def _load_model(self, backend: ComputeBackend, manager: ModelManager) -> object:
        # Detect TFLite by file extension
        model_path = self._model_path or manager.get_path(ModelID.YOLO_V12_N)
        self._is_tflite = str(model_path).endswith(".tflite")
        self._input_quant = None
        # INT8 models (QNN delegate) expect uint8 input, float32 for FP32/FP16
        self._output_quant = None
        if self._is_tflite:
            if "w8a8" in str(model_path) or "int8" in str(model_path):
                self._input_dtype = np.uint8
                # Read quantization info from TFLite model
                try:
                    import tflite_runtime.interpreter as tflite
                except ImportError:
                    import tensorflow.lite as tflite
                interpreter = tflite.Interpreter(model_path=str(model_path))
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()[0]
                scale = input_details.get("quantization", (1.0, 0))[0]
                zero_point = input_details.get("quantization", (1.0, 0))[1]
                dtype = input_details.get("dtype", np.uint8)
                self._input_quant = {"scale": scale, "zero_point": zero_point, "dtype": dtype}
                logger.info(
                    "[YOLO NPU] Quantization params: scale=%.6f, zero_point=%d, dtype=%s",
                    scale,
                    zero_point,
                    dtype,
                )
                # Output quantization
                self._output_quant = self._get_output_quant(str(model_path))
            else:
                self._input_dtype = np.float32
                self._input_quant = None
                self._output_quant = None
            self._input_shape = (1, 640, 640, 3)  # NHWC
        else:
            self._input_dtype = np.float32
            self._input_shape = (1, 3, 640, 640)  # NCHW
            self._input_quant = None
            self._output_quant = None
        logger.info(
            "[YOLO NPU] Model input dtype: %s, shape: %s",
            self._input_dtype,
            self._input_shape,
        )
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
        for gt_det in gt_detections:
            img_path = image_map.get(gt_det.image_name)
            if img_path is None:
                continue
            pred = self._predict_image(img_path, gt_det, handle, backend)
            predictions.append(pred)

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
    input_quant: dict | None = None,
) -> np.ndarray:
    """Load a PIL image and convert to a tensor matching input_shape and quantization."""
    is_nhwc = len(input_shape) == _INPUT_NDIM and input_shape[_NHWC_CHANNEL_AXIS] == _RGB_CHANNELS
    is_nchw = len(input_shape) == _INPUT_NDIM and input_shape[_CHANNEL_AXIS] == _RGB_CHANNELS
    with Image.open(img_path).convert("RGB") as image:
        img_rgb = np.asarray(image, dtype=np.uint8)
    img_bgr = img_rgb[:, :, ::-1]

    if is_nchw:
        _, _, height, width = input_shape
        arr = _preprocess_nchw(img_bgr, height, width)
    elif is_nhwc:
        _, height, width, _ = input_shape
        arr = _preprocess_nhwc(img_bgr, height, width)
    else:
        msg = f"Unsupported input shape for YOLO tensor: {input_shape}"
        raise ValueError(msg)
    logger.info(
        "[YOLO PREPROCESS] Letterbox preprocess applied: shape=%s dtype=%s",
        arr.shape,
        arr.dtype,
    )
    # Quantize if quantization info is provided
    if input_quant is not None:
        scale = input_quant.get("scale", 1.0)
        zero_point = input_quant.get("zero_point", 0)
        dtype = input_quant.get("dtype", np.uint8)
        arr = np.clip(np.round(arr / scale + zero_point), 0, 255).astype(dtype)
        logger.info(
            "[YOLO NPU] Quantized input: min=%s, max=%s, dtype=%s",
            arr.min(),
            arr.max(),
            arr.dtype,
        )
    else:
        logger.info(
            "[YOLO NPU] Float input: min=%s, max=%s, dtype=%s",
            arr.min(),
            arr.max(),
            arr.dtype,
        )
    return arr


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


def _parse_yolo_boxes(  # noqa: C901, PLR0911, PLR0912, PLR0915
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
    import logging

    logger = logging.getLogger(__name__)
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
            logger.info(
                "[YOLO PARSE] 3-tensor: candidate boxes before mask: %d, after mask: %d",
                len(boxes_arr[0]),
                len(boxes_xyxy),
            )
            if len(boxes_xyxy) == 0:
                logger.info("[YOLO PARSE] No boxes after confidence threshold.")
                return []

            keep = _nms_numpy(boxes_xyxy, scores)
            logger.info("[YOLO PARSE] 3-tensor: boxes after NMS: %d", len(keep))
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
        logger.warning("[YOLO PARSE] Output is not ndarray: %r", type(arr))
        return []

    if arr.ndim == _OUTPUT_NDIM and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != _MATRIX_NDIM:
        logger.warning("[YOLO PARSE] Unexpected output ndim for combined format: %s", arr.ndim)
        return []
    if arr.shape[0] == _YOLO_FEATURE_DIM and arr.shape[1] != _YOLO_FEATURE_DIM:
        arr = arr.T
        logger.info("[YOLO PARSE] Transposed output to (N, 84): %s", arr.shape)
    elif arr.shape[1] == _YOLO_FEATURE_DIM and arr.shape[0] != _YOLO_FEATURE_DIM:
        logger.info("[YOLO PARSE] Output already (N, 84): %s", arr.shape)
    else:
        logger.warning("[YOLO PARSE] Unexpected output shape: %s", arr.shape)
        return []

    if arr.shape[0] == 0 or arr.shape[1] <= _BBOX_COORDS:
        logger.info("[YOLO PARSE] No candidate boxes after shape check.")
        return []

    logger.info(
        "[YOLO PARSE] Combined raw stats overall: min=%.6f max=%.6f mean=%.6f p99=%.6f",
        float(arr.min()),
        float(arr.max()),
        float(arr.mean()),
        float(np.percentile(arr, 99)),
    )

    boxes_xywh = arr[:, :_BBOX_COORDS]
    logger.info(
        "[YOLO PARSE] Combined raw box stats [:4]: min=%.6f max=%.6f mean=%.6f p99=%.6f",
        float(boxes_xywh.min()),
        float(boxes_xywh.max()),
        float(boxes_xywh.mean()),
        float(np.percentile(boxes_xywh, 99)),
    )

    # Two common head layouts:
    #   85 channels: [cx, cy, w, h, obj, cls0..cls79]
    #   84 channels: [cx, cy, w, h, cls0..cls79] (objectness fused)
    if arr.shape[1] == _YOLO_FEATURE_DIM + 1:
        conf_raw = arr[:, _BBOX_COORDS]
        class_scores_raw = arr[:, _BBOX_COORDS + 1 :]
        logger.info(
            "[YOLO PARSE] Raw ch4 (obj/conf) stats: min=%.6f max=%.6f mean=%.6f p99=%.6f",
            float(conf_raw.min()),
            float(conf_raw.max()),
            float(conf_raw.mean()),
            float(np.percentile(conf_raw, 99)),
        )
        logger.info(
            "[YOLO PARSE] Raw class stats [5:]: min=%.6f max=%.6f mean=%.6f p99=%.6f",
            float(class_scores_raw.min()),
            float(class_scores_raw.max()),
            float(class_scores_raw.mean()),
            float(np.percentile(class_scores_raw, 99)),
        )

        obj = (
            np.clip(conf_raw, 0.0, 1.0)
            if _looks_like_probability(conf_raw)
            else 1 / (1 + np.exp(-conf_raw))
        )
        class_scores = (
            np.clip(class_scores_raw, 0.0, 1.0)
            if _looks_like_probability(class_scores_raw)
            else 1 / (1 + np.exp(-class_scores_raw))
        )
        if _looks_like_probability(conf_raw) and _looks_like_probability(class_scores_raw):
            logger.info("[YOLO PARSE] 85-ch decode: using raw probabilities (no extra sigmoid).")
        else:
            logger.info("[YOLO PARSE] 85-ch decode: applied sigmoid to logit channels.")

        class_max = class_scores.max(axis=1)
        confidences = obj * class_max
        class_ids_raw = class_scores.argmax(axis=1)
    else:
        class_scores_raw = arr[:, _BBOX_COORDS:]
        logger.info(
            "[YOLO PARSE] Raw class stats [4:]: min=%.6f max=%.6f mean=%.6f p99=%.6f",
            float(class_scores_raw.min()),
            float(class_scores_raw.max()),
            float(class_scores_raw.mean()),
            float(np.percentile(class_scores_raw, 99)),
        )
        # Log channel 4 (first class channel) separately to help distinguish between
        # a potential objectness/conf channel and a true class probability channel.
        # If ch4 has a distinctly higher mean/p99 than the class channels that follow,
        # the model likely has a 85-ch layout but was exported as 84.
        ch4 = class_scores_raw[:, 0]
        ch_rest = class_scores_raw[:, 1:] if class_scores_raw.shape[1] > 1 else class_scores_raw
        logger.info(
            "[YOLO PARSE] Ch4 (first non-box channel) stats: min=%.6f max=%.6f mean=%.6f p99=%.6f",
            float(ch4.min()),
            float(ch4.max()),
            float(ch4.mean()),
            float(np.percentile(ch4, 99)),
        )
        logger.info(
            "[YOLO PARSE] Ch5+ (remaining class channels) stats: "
            "min=%.6f max=%.6f mean=%.6f p99=%.6f",
            float(ch_rest.min()),
            float(ch_rest.max()),
            float(ch_rest.mean()),
            float(np.percentile(ch_rest, 99)),
        )

        class_scores = _activate_maybe_logits(class_scores_raw)
        confidences = class_scores.max(axis=1)
        class_ids_raw = class_scores.argmax(axis=1)
        logger.info(
            "[YOLO PARSE] 84-ch decode: using fused class-prob path across all 80 channels."
        )

    logger.info(
        "[YOLO PARSE] Combined class-prob stats: min=%.6f max=%.6f mean=%.6f p99=%.6f",
        float(class_scores.min()),
        float(class_scores.max()),
        float(class_scores.mean()),
        float(np.percentile(class_scores, 99)),
    )
    logger.info(
        "[YOLO PARSE] Combined confidence stats: min=%.6f max=%.6f mean=%.6f p99=%.6f",
        float(confidences.min()),
        float(confidences.max()),
        float(confidences.mean()),
        float(np.percentile(confidences, 99)),
    )
    logger.info("[YOLO PARSE] Effective conf_threshold=%.6f", conf_threshold)

    mask = confidences >= conf_threshold
    mask_count = int(np.count_nonzero(mask))
    mask_ratio = mask_count / max(1, arr.shape[0])
    logger.info(
        "[YOLO PARSE] Combined: candidate boxes before mask: %d, after mask: %d",
        arr.shape[0],
        mask_count,
    )
    if mask_ratio > _HIGH_PASS_RATIO:
        logger.warning(
            "[YOLO PARSE] %.2f%% of candidates pass conf_threshold=%.3f. "
            "This often indicates score calibration mismatch (e.g., applying sigmoid twice).",
            100 * mask_ratio,
            conf_threshold,
        )
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids_raw = class_ids_raw[mask]

    if len(boxes_xywh) == 0:
        topk = min(_EMPTY_MASK_DEBUG_TOPK, len(confidences))
        if topk > 0:
            # Summarize best candidates when thresholding eliminates all boxes.
            top_idx = np.argsort(confidences)[-topk:][::-1]
            for rank, idx in enumerate(top_idx, start=1):
                class_id = int(class_ids_raw[idx])
                label = str(class_id)
                if class_labels is not None and class_id < len(class_labels):
                    label = class_labels[class_id]
                logger.info(
                    "[YOLO PARSE] Empty-mask top%02d cand idx=%d conf=%.6f class_id=%d label=%s",
                    rank,
                    int(idx),
                    float(confidences[idx]),
                    class_id,
                    label,
                )
            logger.info(
                "[YOLO PARSE] Suggest trying --conf-threshold below %.6f for diagnostic runs.",
                float(confidences[top_idx[-1]]),
            )
        logger.info("[YOLO PARSE] No boxes after confidence threshold.")
        return []

    x1s = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1s = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2s = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2s = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy_raw = np.stack([x1s, y1s, x2s, y2s], axis=1)

    keep = _nms_numpy(boxes_xyxy_raw, confidences)
    logger.info("[YOLO PARSE] Combined: boxes after NMS: %d", len(keep))
    return _build_oracle_boxes(
        boxes_xyxy_raw[keep], confidences[keep], class_ids_raw[keep], img_w, img_h, class_labels
    )
