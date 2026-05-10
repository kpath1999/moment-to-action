"""Preprocess COCO val2017 images into raw float32 files for NPU inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

COCO_VAL_DIR = Path("~/coco/images/val2017").expanduser()
RAW_OUTPUT_DIR = Path("~/coco_raw_inputs").expanduser()
INPUT_LIST_PATH = Path("~/input_list_coco.txt").expanduser()

# Limit to first N images for a quick test — set to 5000 for full eval
MAX_IMAGES = 100

RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_files = sorted([p.name for p in COCO_VAL_DIR.iterdir() if p.suffix == ".jpg"])[:MAX_IMAGES]

input_paths = []
image_ids = []

for img_file in image_files:
    img_path = COCO_VAL_DIR / img_file
    img = Image.open(img_path).convert("RGB").resize((640, 640))
    arr = np.array(img, dtype=np.float32) / 255.0  # normalize to [0,1]
    # Convert HWC → CHW (NCHW for YOLOv6)
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # (1,3,640,640)

    raw_name = img_file.replace(".jpg", ".raw")
    raw_path = RAW_OUTPUT_DIR / raw_name
    arr.tofile(raw_path)

    input_paths.append(raw_path)
    image_ids.append(Path(img_file).stem)  # e.g. "000000118113"

with INPUT_LIST_PATH.open("w") as f:
    f.writelines(f"{p}\n" for p in input_paths)

# Save image id order for later mAP computation
with Path("~/coco_image_ids.txt").expanduser().open("w") as f:
    f.writelines(id_ + "\n" for id_ in image_ids)

print(f"Preprocessed {len(input_paths)} images → {RAW_OUTPUT_DIR}")
print(f"Input list → {INPUT_LIST_PATH}")
