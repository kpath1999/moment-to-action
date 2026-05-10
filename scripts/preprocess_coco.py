"""Preprocess COCO val2017 images into raw float32 files for YOLOv6/QNN inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

COCO_VAL_DIR = Path("~/coco/images/val2017").expanduser()
RAW_OUTPUT_DIR = Path("~/coco_raw_inputs").expanduser()
INPUT_LIST_PATH = Path("~/input_list_coco.txt").expanduser()
IMAGE_IDS_FILE = Path("~/coco_image_ids.txt").expanduser()
META_FILE = Path("~/coco_preprocess_meta.json").expanduser()

MAX_IMAGES = 100
TARGET_SIZE = 640
PAD_VALUE = 114

RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def letterbox_pil(
    img: Image.Image, new_shape: int = 640, color: int = 114
) -> tuple[Image.Image, dict]:
    """Resize image with letterboxing and return canvas with padding metadata."""
    w0, h0 = img.size
    r = min(new_shape / w0, new_shape / h0)
    new_w, new_h = round(w0 * r), round(h0 * r)

    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (new_shape, new_shape), (color, color, color))

    pad_x = (new_shape - new_w) / 2.0
    pad_y = (new_shape - new_h) / 2.0
    left = round(pad_x - 0.1)
    top = round(pad_y - 0.1)
    canvas.paste(resized, (left, top))

    meta = {
        "orig_w": w0,
        "orig_h": h0,
        "ratio": r,
        "pad_x": left,
        "pad_y": top,
        "resized_w": new_w,
        "resized_h": new_h,
    }
    return canvas, meta


image_files = sorted([p.name for p in COCO_VAL_DIR.iterdir() if p.suffix.lower() == ".jpg"])[
    :MAX_IMAGES
]

input_paths = []
image_ids = []
meta_by_id = {}

for img_file in image_files:
    image_id = Path(img_file).stem
    img_path = COCO_VAL_DIR / img_file

    img = Image.open(img_path).convert("RGB")
    img_lb, meta = letterbox_pil(img, TARGET_SIZE, PAD_VALUE)

    arr = np.array(img_lb, dtype=np.float32)
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = arr[np.newaxis, ...]  # -> NCHW
    arr = np.ascontiguousarray(arr)

    raw_path = RAW_OUTPUT_DIR / f"{image_id}.raw"
    arr.tofile(raw_path)

    input_paths.append(raw_path)
    image_ids.append(image_id)
    meta_by_id[image_id] = meta | {"file_name": img_file}

with INPUT_LIST_PATH.open("w") as f:
    f.writelines(f"{p}\n" for p in input_paths)

with IMAGE_IDS_FILE.open("w") as f:
    f.writelines(f"{image_id}\n" for image_id in image_ids)

with META_FILE.open("w") as f:
    json.dump(meta_by_id, f, indent=2)

print(f"Preprocessed {len(input_paths)} images → {RAW_OUTPUT_DIR}")
print(f"Input list → {INPUT_LIST_PATH}")
print(f"Metadata → {META_FILE}")
