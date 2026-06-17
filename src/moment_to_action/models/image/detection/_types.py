"""Detection output types for image-based object detectors."""

from __future__ import annotations

import attrs

# The 80 COCO object-detection class names, in label-index order.  Shared by
# detectors that produce COCO-class predictions (e.g. the Detectron2 detector).
COCO_LABELS: tuple[str, ...] = (
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


@attrs.frozen
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates.

    Attributes:
        x1: Left edge in pixels.
        y1: Top edge in pixels.
        x2: Right edge in pixels.
        y2: Bottom edge in pixels.
    """

    x1: float
    y1: float
    x2: float
    y2: float


@attrs.frozen
class Detection:
    """Single object detection result.

    Attributes:
        label: Human-readable class name.
        confidence: Detection confidence in [0, 1].
        bbox: Bounding box in original image pixel coordinates.
    """

    label: str
    confidence: float
    bbox: BoundingBox
