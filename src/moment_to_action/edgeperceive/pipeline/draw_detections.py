"""draw_detections.py

Runs the YOLO pipeline and draws bounding boxes on the image.

Usage:
    uv run python draw_detections.py --image weapon.jpg --model model.onnx
    uv run python draw_detections.py --image weapon.jpg --model model.onnx --conf 0.3 --out result.jpg
"""

import argparse
import logging
import sys
import time

import cv2

from moment_to_action.edgeperceive.core.messages import DetectionMessage, RawFrameMessage
from moment_to_action.edgeperceive.hardware.types import ComputeUnit
from moment_to_action.edgeperceive.stages import (
    Pipeline,
    PreprocessorStage,
    SensorStage,
    YOLOStage,
)
from moment_to_action.edgeperceive.stages.base import Stage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--conf", type=float, default=0.3)
parser.add_argument("--out", default="detections.jpg")
args = parser.parse_args()


# ── intercept DetectionMessage before ReasoningStage ──────────────
class CaptureStage(Stage):
    """Stores the DetectionMessage and passes it through."""

    def __init__(self):
        self.detections = None

    def process(self, msg):
        self.detections = msg
        return msg


capture = CaptureStage()

pipeline = Pipeline(
    [
        SensorStage(),
        PreprocessorStage(target_size=(640, 640), letterbox=True),
        YOLOStage(
            model_path=args.model, confidence_threshold=args.conf, compute_unit=ComputeUnit.CPU
        ),
        capture,
    ]
)

# ── run ────────────────────────────────────────────────────────────
msg = RawFrameMessage(frame=None, timestamp=time.time(), source=args.image)
pipeline.run(msg)

if capture.detections is None:
    print(f"No detections above conf={args.conf}")
    sys.exit(0)

# ── draw ───────────────────────────────────────────────────────────
frame = cv2.imread(args.image)
det: DetectionMessage = capture.detections


# Color per label (consistent across runs)
def label_color(label: str) -> tuple:
    h = abs(hash(label)) % 179
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h / 179.0, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR


for box in sorted(det.boxes, key=lambda b: b.confidence):
    x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
    color = label_color(box.label)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{box.label} {box.confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

cv2.imwrite(args.out, frame)
print(f"\n{len(det.boxes)} detection(s) drawn → {args.out}")
for b in sorted(det.boxes, key=lambda b: -b.confidence):
    print(f"  {b.label:20s}  {b.confidence:.2f}  [{b.x1:.0f},{b.y1:.0f},{b.x2:.0f},{b.y2:.0f}]")
