"""Detectron2 (Faster R-CNN) two-stage object detection model.

Unlike the single-graph detectors (YOLO, RF-DETR, RTMDet), Detectron2 is a
two-component model exported from Qualcomm AI Hub:

1. ``proposal_generator(image)`` -> ``feature``, ``proposals``, ``score`` — the
   backbone + RPN.  Runs on the accelerator.
2. CPU glue (:meth:`Detectron2Model._filter_proposals`) — top-k by objectness,
   drop empty boxes, NMS, truncate + zero-pad to a fixed proposal count.
3. ``roi_head(features, proposals_boxes)`` -> ``boxes``, ``scores``,
   ``classes`` — the ROI pooling + box head.  Runs on the accelerator.
4. CPU post-processing (:meth:`Detectron2Model.post_proc`) — per-class NMS +
   scale to original image coordinates.

Both graphs quantize to full integer precision (``w8a8`` / ``w8a16``), so the
context binaries have integer I/O and link on Hexagon v68 (QCS6490) — the
reason this detector supports the NPU where RF-DETR / RTMDet do not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import cv2
import numpy as np

from moment_to_action.models._artifacts import resolve_backend_artifact
from moment_to_action.models._formats import ModelFormat
from moment_to_action.models.image.detection._base import ImageDetectionModel
from moment_to_action.models.image.detection._types import (
    COCO_LABELS,
    BoundingBox,
    Detection,
)

if TYPE_CHECKING:
    from pathlib import Path

    from moment_to_action.hardware import ComputeBackend

_INPUT_SIZE = 800
_PG_STEM = "model.proposal_generator"
_ROI_STEM = "model.roi_head"
_MAX_PRE_NMS = 6000
_MAX_POST_NMS = 200
_PROPOSAL_IOU = 0.7
_BOX_IOU = 0.5


def _letterbox_params(orig_h: int, orig_w: int) -> tuple[float, int, int]:
    """Aspect-preserving resize-and-pad parameters for an ``_INPUT_SIZE`` square.

    Mirrors qai_hub_models' ``resize_pad`` (center float): scale by the smaller
    ratio, then center the resized image in the square canvas.

    Args:
        orig_h: Original frame height.
        orig_w: Original frame width.

    Returns:
        ``(scale, pad_left, pad_top)``.  A model-space box maps back to the
        original via ``(coord - pad) / scale``.
    """
    scale = min(_INPUT_SIZE / orig_h, _INPUT_SIZE / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    pad_left = (_INPUT_SIZE - new_w) // 2
    pad_top = (_INPUT_SIZE - new_h) // 2
    return scale, pad_left, pad_top


class Detectron2Model(ImageDetectionModel):
    """Detectron2 Faster R-CNN object detector (COCO 80-class).

    A two-stage detector with two on-device graphs (proposal generator + ROI
    head) and CPU glue between and after them.  Supports ONNX (CPU/GPU) and DLC
    (NPU via QAIRT) formats.  The model is unloaded after construction; call
    :meth:`load` before inference.

    Both component artifacts live side by side in a single variant directory,
    resolved by stem: ``model.proposal_generator.*`` and ``model.roi_head.*``.

    Args:
        variant: Registry variant key used to identify this instance.
        path: Directory containing the component artifacts.
        model_format: Model file format — determines which backend methods to call.
        confidence_threshold: Minimum confidence score to keep a detection.
    """

    COCO_LABELS: ClassVar[tuple[str, ...]] = COCO_LABELS

    def __init__(
        self,
        variant: str,
        path: Path,
        model_format: ModelFormat,
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initialize an unloaded Detectron2Model.

        Args:
            variant: Registry variant key.
            path: Directory holding ``model.proposal_generator.*`` and
                ``model.roi_head.*`` artifacts.
            model_format: ``ModelFormat.ONNX`` or ``ModelFormat.DLC``.
            confidence_threshold: Detections below this score are discarded.
        """
        super().__init__(variant, path)
        self._format = model_format
        self._confidence_threshold = confidence_threshold
        # ONNX is the single-graph float export (detectron2 tracing): one
        # end-to-end model.onnx that runs RPN + ROI head + NMS internally.  DLC
        # is the two-component AI Hub split (proposal generator + ROI head).
        self._single_graph_onnx = model_format is ModelFormat.ONNX
        self._handle_pg: object = None
        self._handle_roi: object = None
        # Whether the loaded ROI-head artifact declares its `features` input as
        # channel-last (NHWC).  Only the HTP context binary (.npu.bin) does; the
        # portable float .dlc keeps NCHW.  Set in load(); gates the transpose in
        # run().  ONNX leaves this False (NCHW features).
        self._roi_channel_last: bool = False
        self._last_original_size: tuple[int, int] | None = None
        # AI Hub qcs6490 DLC exports to NHWC; all other variants use NCHW.  Both
        # precision variants ("qcs6490_w8a16", "qcs6490_w8a8") share the layout.
        self._input_layout = (
            "NHWC"
            if (model_format is ModelFormat.DLC and variant.startswith("qcs6490"))
            else "NCHW"
        )

    @property
    def confidence_threshold(self) -> float:
        """Minimum confidence score kept by :meth:`post_proc`."""
        return self._confidence_threshold

    @property
    def input_layout(self) -> str:
        """Input tensor layout: ``"NCHW"`` or ``"NHWC"``."""
        return self._input_layout

    def load(self, backend: ComputeBackend) -> None:
        """Load the model graph(s) onto the backend.

        For the single-graph ONNX variant, loads ``model.onnx`` from the variant
        directory.  For DLC variants, resolves each component via
        :func:`~moment_to_action.models._artifacts.resolve_backend_artifact`
        with its stem — a per-backend context binary (``<stem>.npu.bin``) when
        present, falling back to ``<stem>.dlc``.

        Args:
            backend: Hardware backend to load the model onto.

        Raises:
            RuntimeError: If the model is already loaded.
        """
        if self._backend is not None:
            msg = f"{type(self).__name__} is already loaded; call unload() first"
            raise RuntimeError(msg)
        if self._single_graph_onnx:
            # Single end-to-end graph; reuse _handle_pg as the sole handle.
            self._handle_pg = backend.load_model(self._path / "model.onnx")
        else:
            pg = resolve_backend_artifact(self._path, backend.preferred_unit, stem=_PG_STEM)
            roi = resolve_backend_artifact(self._path, backend.preferred_unit, stem=_ROI_STEM)
            # Only the HTP context binary lays `features` out channel-last; the
            # float .dlc keeps NCHW, so the transpose in run() is binary-only.
            self._roi_channel_last = roi.name.endswith(".npu.bin")
            self._handle_pg = backend.load_model_dlc(pg)
            self._handle_roi = backend.load_model_dlc(roi)
        self._backend = backend

    def unload(self) -> None:
        """Release backend resources for the loaded graph(s) and reset state."""
        if self._backend is not None:
            if self._single_graph_onnx:
                self._backend.unload_model(self._handle_pg)
            else:
                self._backend.unload_dlc(self._handle_pg)
                self._backend.unload_dlc(self._handle_roi)
        self._backend = None
        self._handle_pg = None
        self._handle_roi = None

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        """Letterbox and format a raw BGR frame for Detectron2 inference.

        Both paths aspect-preserve resize + center pad into an ``_INPUT_SIZE``
        square (matching the Detectron2 app's ``resize_pad``); the decode undoes
        the pad/scale.  Formatting then differs by graph:

        Args:
            frame: Raw BGR image (HxWxC, uint8).

        Returns:
            Float32 tensor:

            - ``(3, 800, 800)`` raw **BGR, 0-255** for the single-graph ONNX
              (detectron2 normalizes with PIXEL_MEAN/STD internally; no batch dim).
            - ``(1, 3, 800, 800)`` **RGB, [0, 1]** when :attr:`input_layout` is
              ``"NCHW"`` (DLC).
            - ``(1, 800, 800, 3)`` **RGB, [0, 1]** when :attr:`input_layout` is
              ``"NHWC"`` (qcs6490 DLC).
        """
        orig_h, orig_w = frame.shape[0], frame.shape[1]
        self._last_original_size = (orig_h, orig_w)
        scale, pad_left, pad_top = _letterbox_params(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=frame.dtype)
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        if self._single_graph_onnx:
            # detectron2 tracing contract: raw BGR, 0-255 float32, CHW, no batch.
            return canvas.astype(np.float32).transpose(2, 0, 1)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        if self._input_layout == "NHWC":
            return np.expand_dims(normalized, axis=0)
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def run(self, prepared: np.ndarray) -> list[np.ndarray]:
        """Run the Detectron2 forward pass.

        The single-graph ONNX runs the whole detector end-to-end (RPN + ROI head
        + NMS).  The DLC path runs the proposal generator, filters proposals on
        the CPU, then runs the ROI head on the filtered proposals.

        Args:
            prepared: Tensor from :meth:`prepare`.

        Returns:
            List of raw output tensors ``[boxes, scores, classes]``, each with a
            leading batch dim, ready for :meth:`post_proc`.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._backend is None:
            msg = "Detectron2Model.load() must be called before run()"
            raise RuntimeError(msg)

        if self._single_graph_onnx:
            # Traced GeneralizedRCNN outputs (in order): boxes [N,4], classes
            # [N] int64, scores [N], image_size [2].  Reorder to [boxes, scores,
            # classes] and add the batch dim _decode expects.
            outs = self._backend.run(self._handle_pg, prepared)
            boxes, classes, scores = outs[0], outs[1], outs[2]
            return [boxes[np.newaxis], scores[np.newaxis], classes[np.newaxis]]

        out1 = self._backend.infer_dlc(self._handle_pg, prepared)
        padded = self._filter_proposals(out1["proposals"], out1["score"])
        # The proposal generator emits `feature` as NCHW.  Only the HTP context
        # binary declares the ROI head's `features` input as channel-last
        # (get_channel_last_inputs=["features"]) -- qai_hub's on-device wrapper
        # auto-transposes there, infer_dlc does not, so we transpose NCHW ->
        # NHWC for the binary.  The portable float .dlc keeps NCHW; transposing
        # it would pool garbage features.
        if self._roi_channel_last:
            feat = np.ascontiguousarray(np.transpose(out1["feature"], (0, 2, 3, 1)))
        else:
            feat = np.ascontiguousarray(out1["feature"])
        out2 = self._backend.infer_dlc(
            self._handle_roi,
            {"features": feat, "proposals_boxes": padded},
        )
        boxes, scores, classes = out2["boxes"], out2["scores"], out2["classes"]
        return [boxes, scores, classes]

    def _filter_proposals(self, proposals: np.ndarray, score: np.ndarray) -> np.ndarray:
        """Select and pad region proposals for the ROI head (batch size 1).

        Clamps proposals to the image, keeps the top ``_MAX_PRE_NMS`` by
        objectness, drops empty boxes, applies NMS at ``_PROPOSAL_IOU``,
        truncates to ``_MAX_POST_NMS`` and zero-pads to that fixed length so the
        ROI head sees a static-shape input.

        Args:
            proposals: ``[1, N, 4]`` proposal boxes (x1, y1, x2, y2) in 800x800 space.
            score: ``[1, N]`` objectness logits.

        Returns:
            ``[1, _MAX_POST_NMS, 4]`` float32 padded proposal boxes.
        """
        boxes = proposals[0].astype(np.float32).copy()
        scores = score[0].astype(np.float32)

        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, float(_INPUT_SIZE))
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, float(_INPUT_SIZE))

        top_k = min(scores.shape[0], _MAX_PRE_NMS)
        topk_idx = np.argsort(scores)[::-1][:top_k]
        boxes = boxes[topk_idx]
        scores = scores[topk_idx]

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        nonempty = (widths > 0) & (heights > 0)
        boxes = boxes[nonempty]
        scores = scores[nonempty]

        keep = self._nms(boxes, scores, iou_threshold=_PROPOSAL_IOU)[:_MAX_POST_NMS]
        selected = boxes[keep]

        padded = np.zeros((_MAX_POST_NMS, 4), dtype=np.float32)
        padded[: len(selected)] = selected
        return padded[np.newaxis, :, :]

    def post_proc(self, raw: list[np.ndarray]) -> list[Detection]:
        """Decode ROI-head outputs into detections.

        Args:
            raw: Value returned by :meth:`run`.

        Returns:
            Detections above :attr:`confidence_threshold` after per-class NMS,
            scaled to the original frame's pixel coordinates using the size
            recorded by the preceding :meth:`prepare` call.
        """
        return self._decode(raw, original_size=self._last_original_size)

    def decode(self, raw: object, original_size: tuple[int, int]) -> list[Detection]:
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
        """Filter, per-class NMS, and scale ROI-head outputs.

        Args:
            outputs: ``[boxes (1,N,4), scores (1,N), classes (1,N)]``.
            original_size: ``(height, width)`` to scale boxes to; ``None`` keeps 800x800 coords.

        Returns:
            Filtered, NMS-reduced, scaled list of detections.
        """
        _expected_output_count = 3
        if len(outputs) < _expected_output_count:
            return []

        boxes = outputs[0][0].astype(np.float32)  # [N, 4]
        scores = outputs[1][0].astype(np.float32)  # [N]
        classes = outputs[2][0].astype(np.int64)  # [N]

        mask = scores >= self._confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]

        if len(boxes) == 0:
            return []

        keep = self._per_class_nms(boxes, scores, classes, iou_threshold=_BOX_IOU)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

        if original_size is not None:
            orig_h, orig_w = original_size
            scale, pad_left, pad_top = _letterbox_params(orig_h, orig_w)
        else:
            scale, pad_left, pad_top = 1.0, 0, 0
            orig_w, orig_h = _INPUT_SIZE, _INPUT_SIZE

        # Undo the letterbox: subtract pad, divide by scale, clamp to the frame.
        detections: list[Detection] = []
        for box, score, cid in zip(boxes, scores, classes, strict=False):
            x1 = min(max(0.0, (float(box[0]) - pad_left) / scale), float(orig_w))
            y1 = min(max(0.0, (float(box[1]) - pad_top) / scale), float(orig_h))
            x2 = min(max(0.0, (float(box[2]) - pad_left) / scale), float(orig_w))
            y2 = min(max(0.0, (float(box[3]) - pad_top) / scale), float(orig_h))
            class_id = int(cid)
            in_range = 0 <= class_id < len(self.COCO_LABELS)
            label = self.COCO_LABELS[class_id] if in_range else str(class_id)
            detections.append(
                Detection(
                    label=label,
                    confidence=float(score),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return detections

    @classmethod
    def _per_class_nms(
        cls,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        iou_threshold: float,
    ) -> list[int]:
        """Run NMS independently within each class.

        Args:
            boxes: ``[N, 4]`` float32 boxes.
            scores: ``[N]`` float32 scores.
            classes: ``[N]`` int class indices.
            iou_threshold: IoU threshold for suppression.

        Returns:
            Indices into ``boxes`` to keep, sorted by descending score.
        """
        keep: list[int] = []
        for c in np.unique(classes):
            cls_idx = np.nonzero(classes == c)[0]
            local = cls._nms(boxes[cls_idx], scores[cls_idx], iou_threshold)
            keep.extend(int(cls_idx[i]) for i in local)
        keep.sort(key=lambda i: float(scores[i]), reverse=True)
        return keep

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
