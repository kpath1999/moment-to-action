from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003

import attrs
import numpy as np

from moment_to_action.benchmark._oracle_ground_truth import OracleDetection

_PRECISION_ARRAY_NDIM = 5
_IOU_MATCH_THRESHOLD = 0.5


@attrs.frozen
class DetectionMetrics:
    """Detection metrics summary for pseudo-ground-truth evaluation."""

    map_50: float
    map_50_95: float
    recall_50: float
    per_class_ap: dict[str, float]


def compute_detection_map(  # noqa: C901, PLR0912, PLR0915
    predictions: Sequence[OracleDetection],
    ground_truth: Sequence[OracleDetection],
) -> DetectionMetrics:
    """Compute COCO-style mAP against oracle detections.

    Args:
        predictions: Predicted detections by image.
        ground_truth: Oracle detections by image.

    Returns:
        DetectionMetrics populated with mAP and recall summaries.
    """
    gt_by_name = {det.image_name: det for det in ground_truth}
    pred_by_name = {det.image_name: det for det in predictions}
    image_names = sorted(gt_by_name)

    if not image_names:
        return DetectionMetrics(map_50=0.0, map_50_95=0.0, recall_50=0.0, per_class_ap={})

    category_names = sorted(
        {box.label for det in ground_truth for box in det.boxes}
        | {box.label for det in predictions for box in det.boxes}
    )
    if not category_names:
        return DetectionMetrics(map_50=0.0, map_50_95=0.0, recall_50=0.0, per_class_ap={})

    category_id_by_name = {name: idx + 1 for idx, name in enumerate(category_names)}
    image_id_by_name = {name: idx + 1 for idx, name in enumerate(image_names)}

    coco_gt = {
        "images": [{"id": image_id_by_name[name], "file_name": name} for name in image_names],
        "categories": [{"id": category_id_by_name[name], "name": name} for name in category_names],
        "annotations": [],
    }

    annotation_id = 1
    for image_name in image_names:
        for box in gt_by_name[image_name].boxes:
            width = max(0.0, box.x2 - box.x1)
            height = max(0.0, box.y2 - box.y1)
            if width <= 0.0 or height <= 0.0:
                continue
            coco_gt["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id_by_name[image_name],
                    "category_id": category_id_by_name[box.label],
                    "bbox": [box.x1, box.y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco_dt: list[dict[str, float | int | list[float]]] = []
    for image_name, det in pred_by_name.items():
        image_id = image_id_by_name.get(image_name)
        if image_id is None:
            continue
        for box in det.boxes:
            category_id = category_id_by_name.get(box.label)
            if category_id is None:
                continue
            width = max(0.0, box.x2 - box.x1)
            height = max(0.0, box.y2 - box.y1)
            if width <= 0.0 or height <= 0.0:
                continue
            coco_dt.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [box.x1, box.y1, width, height],
                    "score": box.confidence,
                }
            )

    if not coco_gt["annotations"]:
        return DetectionMetrics(map_50=0.0, map_50_95=0.0, recall_50=0.0, per_class_ap={})

    if not coco_dt:
        return DetectionMetrics(map_50=0.0, map_50_95=0.0, recall_50=0.0, per_class_ap={})

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt_api = COCO()
    coco_gt_api.dataset = coco_gt
    coco_gt_api.createIndex()

    coco_dt_api = coco_gt_api.loadRes(coco_dt)
    evaluator = COCOeval(coco_gt_api, coco_dt_api, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    if not isinstance(evaluator.stats, np.ndarray) or evaluator.stats.size == 0:
        return DetectionMetrics(map_50=0.0, map_50_95=0.0, recall_50=0.0, per_class_ap={})

    map_50_95 = float(evaluator.stats[0])
    map_50 = float(evaluator.stats[1])
    recall_50 = _recall_at_iou_50(predictions=predictions, ground_truth=ground_truth)

    per_class_ap: dict[str, float] = {}
    precision = evaluator.eval.get("precision")
    if isinstance(precision, np.ndarray) and precision.ndim == _PRECISION_ARRAY_NDIM:
        iou_index = 0  # evaluator.params.iouThrs starts at 0.5 by default
        for class_index, category_name in enumerate(category_names):
            class_precision = precision[iou_index, :, class_index, 0, -1]
            valid = class_precision[class_precision > -1]
            per_class_ap[category_name] = float(np.mean(valid)) if valid.size > 0 else 0.0

    return DetectionMetrics(
        map_50=map_50,
        map_50_95=map_50_95,
        recall_50=recall_50,
        per_class_ap=per_class_ap,
    )


def _recall_at_iou_50(
    predictions: Sequence[OracleDetection],
    ground_truth: Sequence[OracleDetection],
) -> float:
    pred_by_name = {det.image_name: det for det in predictions}

    matched = 0
    total = 0
    for gt_det in ground_truth:
        pred_boxes = pred_by_name.get(
            gt_det.image_name,
            OracleDetection(gt_det.image_name, []),
        ).boxes
        for gt_box in gt_det.boxes:
            total += 1
            candidates = [pb for pb in pred_boxes if pb.label == gt_box.label]
            best_iou = max((gt_box.iou(pb) for pb in candidates), default=0.0)
            if best_iou >= _IOU_MATCH_THRESHOLD:
                matched += 1

    if total == 0:
        return 0.0
    return matched / total
