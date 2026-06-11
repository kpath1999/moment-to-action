"""Debug script: inspect raw DLC output tensors for YOLO qcs6490 variant."""

from __future__ import annotations

import sys

import cv2

from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.models import ModelID, ModelManager
from moment_to_action.paths import PathManager

img_path = sys.argv[1] if len(sys.argv) > 1 else "images/pedestrian.jpg"

frame = cv2.imread(img_path)
if frame is None:
    print(f"Could not read: {img_path}")
    sys.exit(1)

path_mgr = PathManager()
model = ModelManager(path_mgr).get_model(ModelID.YOLO_V8, variant="qcs6490")
backend = ComputeBackend(ComputeUnit.CPU)
model.load(backend)

prepared = model.prepare(frame)
dlc_out = backend.infer_dlc(model._handle, prepared)  # type: ignore[attr-defined]  # noqa: SLF001

print("=== DLC output keys ===")
for k, v in dlc_out.items():
    print(f"  {k!r}: shape={v.shape}, dtype={v.dtype}, min={v.min():.4f}, max={v.max():.4f}")

_THRESH_HIGH = 0.1
_THRESH_MID = 0.05
_THRESH_LOW = 0.01

if "cls" in dlc_out:
    cls = dlc_out["cls"]
    scores = cls.max(axis=-1)
    print("\n=== scores (cls.max axis=-1) ===")
    print(f"  shape={scores.shape}, min={scores.min():.4f}, max={scores.max():.4f}")
    hi = (scores > _THRESH_HIGH).sum()
    mid = (scores > _THRESH_MID).sum()
    lo = (scores > _THRESH_LOW).sum()
    print(f"  >{_THRESH_HIGH}: {hi}, >{_THRESH_MID}: {mid}, >{_THRESH_LOW}: {lo}")
else:
    print("\nWARNING: 'cls' key not found in DLC output!")

model.unload()
