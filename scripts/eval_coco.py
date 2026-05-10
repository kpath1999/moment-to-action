"""Evaluate YOLOv6 NPU output against COCO ground truth using pycocotools."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

OUTPUT_DIR = Path("~/output_npu_coco").expanduser()
LABELS_FILE = Path("~/yolov6-qnn_context_binary-w8a16-qualcomm_qcs6490/labels.txt").expanduser()
ANNOTATIONS = Path("~/coco/annotations/instances_val2017.json").expanduser()
IMAGE_IDS_FILE = Path("~/coco_image_ids.txt").expanduser()

with LABELS_FILE.open() as f:
    labels = [line.strip() for line in f]

with IMAGE_IDS_FILE.open() as f:
    image_ids = [int(line.strip()) for line in f]

coco_gt = COCO(ANNOTATIONS)

# Build COCO category name → id map
cat_name_to_id = {cat["name"]: cat["id"] for cat in coco_gt.loadCats(coco_gt.getCatIds())}

results = []

for i, image_id in enumerate(image_ids):
    result_dir = OUTPUT_DIR / f"Result_{i}"

    boxes = np.fromfile(result_dir / "boxes.raw", dtype=np.float32).reshape(-1, 4)
    scores = np.fromfile(result_dir / "scores.raw", dtype=np.float32).reshape(-1)
    class_idx = np.fromfile(result_dir / "class_idx.raw", dtype=np.float32).reshape(-1).astype(int)

    CONF_THRESHOLD = 0.25
    keep = scores > CONF_THRESHOLD

    for box, score, cls in zip(boxes[keep], scores[keep], class_idx[keep], strict=False):
        x1, y1, x2, y2 = box
        label_name = labels[cls] if cls < len(labels) else "unknown"
        cat_id = cat_name_to_id.get(label_name, -1)
        if cat_id == -1:
            continue
        results.append(
            {
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            }
        )

# Save and evaluate
results_path = Path("~/coco_results.json").expanduser()
with results_path.open("w") as f:
    json.dump(results, f)

coco_dt = coco_gt.loadRes(str(results_path))
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.params.imgIds = image_ids
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
