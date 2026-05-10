"""Evaluate YOLOv6 QNN output against COCO and visualize 5 predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

OUTPUT_DIR = Path("~/output_npu_coco").expanduser()
LABELS_FILE = Path("~/yolov6-qnn_context_binary-w8a16-qualcomm_qcs6490/labels.txt").expanduser()
ANNOTATIONS = Path("~/coco/annotations/instances_val2017.json").expanduser()
IMAGE_IDS_FILE = Path("~/coco_image_ids.txt").expanduser()
PREPROCESS_META = Path("~/coco_preprocess_meta.json").expanduser()
COCO_VAL_DIR = Path("~/coco/images/val2017").expanduser()

RESULTS_PATH = Path("~/coco_results.json").expanduser()
VIS_DIR = Path("~/coco_vis").expanduser()
VIS_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.25
MAX_VIS_IMAGES = 5

with LABELS_FILE.open() as f:
    labels = [line.strip() for line in f]

with IMAGE_IDS_FILE.open() as f:
    image_ids = [line.strip() for line in f]

with PREPROCESS_META.open() as f:
    preprocess_meta = json.load(f)

coco_gt = COCO(str(ANNOTATIONS))
cat_name_to_id = {cat["name"]: cat["id"] for cat in coco_gt.loadCats(coco_gt.getCatIds())}

results = []


def clip_box(
    x1: float, y1: float, x2: float, y2: float, w: float, h: float
) -> tuple[float, float, float, float]:
    """Clip box coordinates to image boundaries."""
    x1 = max(0.0, min(float(x1), w - 1))
    y1 = max(0.0, min(float(y1), h - 1))
    x2 = max(0.0, min(float(x2), w - 1))
    y2 = max(0.0, min(float(y2), h - 1))
    return x1, y1, x2, y2


def unletterbox_box(box: list[float], meta: dict) -> tuple[float, float, float, float]:
    """Reverse letterbox padding and scaling to recover original image coordinates."""
    x1, y1, x2, y2 = [float(v) for v in box]
    r = float(meta["ratio"])
    pad_x = float(meta["pad_x"])
    pad_y = float(meta["pad_y"])
    orig_w = int(meta["orig_w"])
    orig_h = int(meta["orig_h"])

    x1 = (x1 - pad_x) / r
    y1 = (y1 - pad_y) / r
    x2 = (x2 - pad_x) / r
    y2 = (y2 - pad_y) / r

    return clip_box(x1, y1, x2, y2, orig_w, orig_h)


def draw_predictions(image_id: str, detections: list[dict], max_draw: int = 20) -> None:
    """Draw top detections onto the original image and save to VIS_DIR."""
    meta = preprocess_meta[image_id]
    img_path = COCO_VAL_DIR / meta["file_name"]
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)[:max_draw]
    for det in detections:
        x, y, w, h = det["bbox"]
        x2 = x + w
        y2 = y + h
        label = f"{det['label']} {det['score']:.2f}"
        draw.rectangle([x, y, x2, y2], outline="red", width=2)
        draw.text((x + 2, max(0, y - 10)), label, fill="yellow", font=font)

    out_path = VIS_DIR / f"{image_id}.jpg"
    image.save(out_path, quality=95)


for i, image_id_str in enumerate(image_ids):
    result_dir = OUTPUT_DIR / f"Result_{i}"
    meta = preprocess_meta[image_id_str]

    boxes = np.fromfile(result_dir / "boxes.raw", dtype=np.float32).reshape(-1, 4)
    scores = np.fromfile(result_dir / "scores.raw", dtype=np.float32).reshape(-1)
    class_idx = np.fromfile(result_dir / "class_idx.raw", dtype=np.float32).reshape(-1).astype(int)

    keep = scores > CONF_THRESHOLD
    image_results = []

    for box, score, cls in zip(boxes[keep], scores[keep], class_idx[keep], strict=False):
        if cls < 0 or cls >= len(labels):
            continue

        label_name = labels[cls]
        cat_id = cat_name_to_id.get(label_name, -1)
        if cat_id == -1:
            continue

        x1, y1, x2, y2 = unletterbox_box(box, meta)
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw < 1 or bh < 1:
            continue

        det = {
            "image_id": int(image_id_str),
            "category_id": cat_id,
            "bbox": [x1, y1, bw, bh],
            "score": float(score),
            "label": label_name,
        }
        results.append({k: det[k] for k in ["image_id", "category_id", "bbox", "score"]})
        image_results.append(det)

    if i < MAX_VIS_IMAGES:
        draw_predictions(image_id_str, image_results)

with RESULTS_PATH.open("w") as f:
    json.dump(results, f)

coco_dt = coco_gt.loadRes(str(RESULTS_PATH))
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.params.imgIds = [int(x) for x in image_ids]
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print(f"Saved COCO results to {RESULTS_PATH}")
print(f"Saved {min(MAX_VIS_IMAGES, len(image_ids))} visualizations to {VIS_DIR}")
