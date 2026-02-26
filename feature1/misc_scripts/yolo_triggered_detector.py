import cv2
import numpy as np
import onnxruntime as ort
import time
import logging
import os
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Dequantization constants matching your quantized model export
BOX_ZERO_POINT = 36
BOX_SCALE      = 3.26531720161438
SCORE_SCALE    = 0.00390625   # 1/256


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple        # (x1, y1, x2, y2) in original frame pixel coords
    frame_index: int


@dataclass
class SamplingConfig:
    video_path: str
    model_path: str
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    input_size: tuple = (640, 640)
    classes_of_interest: Optional[list] = None   # None = all; [0] = person only
    save_annotated_frames: bool = True
    output_dir: str = "detections"
    use_qnn: bool = False   # set True to use QNN/NPU provider


def _nms(boxes, scores, iou_threshold):
    """Pure numpy NMS."""
    indices = np.argsort(scores)[::-1]
    keep = []
    while len(indices) > 0:
        cur = indices[0]
        keep.append(cur)
        if len(indices) == 1:
            break
        cb = boxes[cur]
        rb = boxes[indices[1:]]
        x1 = np.maximum(cb[0], rb[:, 0])
        y1 = np.maximum(cb[1], rb[:, 1])
        x2 = np.minimum(cb[2], rb[:, 2])
        y2 = np.minimum(cb[3], rb[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        cur_area  = (cb[2] - cb[0]) * (cb[3] - cb[1])
        rem_areas = (rb[:, 2] - rb[:, 0]) * (rb[:, 3] - rb[:, 1])
        iou = inter / (cur_area + rem_areas - inter + 1e-6)
        indices = indices[1:][iou < iou_threshold]
    return keep


def _postprocess(outputs, orig_shape, frame_idx, iw, ih, config):
    # Your model has 3 separate outputs: boxes (uint8), scores (uint8), classes (int)
    boxes   = (outputs[0].astype(np.float32) - BOX_ZERO_POINT) * BOX_SCALE  # (1,N,4)
    scores  = outputs[1].astype(np.float32) * SCORE_SCALE                   # (1,N)
    classes = outputs[2]                                                     # (1,N)

    boxes   = boxes[0]
    scores  = scores[0]
    classes = classes[0]

    mask = scores >= config.conf_threshold
    boxes, scores, classes = boxes[mask], scores[mask], classes[mask]

    if config.classes_of_interest is not None:
        coi = np.isin(classes, config.classes_of_interest)
        boxes, scores, classes = boxes[coi], scores[coi], classes[coi]

    if len(boxes) == 0:
        return []

    keep = _nms(boxes, scores, config.iou_threshold)
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

    oh, ow = orig_shape[:2]
    sx, sy = ow / iw, oh / ih

    detections = []
    for box, score, cls in zip(boxes, scores, classes):
        x1 = max(0,  int(box[0] * sx))
        y1 = max(0,  int(box[1] * sy))
        x2 = min(ow, int(box[2] * sx))
        y2 = min(oh, int(box[3] * sy))
        cid  = int(cls)
        name = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else str(cid)
        detections.append(Detection(
            class_id=cid, class_name=name, confidence=float(score),
            bbox=(x1, y1, x2, y2), frame_index=frame_idx,
        ))
    return detections


def _draw_detections(frame, detections):
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def run_yolo_on_trigger(config: SamplingConfig):
    """
    Call directly from your YAMNet trigger handler.
    Runs synchronously — no background threads, no TFLite/NPU conflicts.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 1. Grab first frame
    logger.info(f"[1/3] Opening video: {config.video_path}")
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {config.video_path}")
        return []
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        logger.error("Failed to read first frame.")
        return []
    logger.info(f"[1/3] Frame grabbed. Shape: {frame.shape}")

    # 2. Load ONNX model
    logger.info(f"[2/3] Loading ONNX model: {config.model_path}")
    try:
        providers = (
            [('QNNExecutionProvider', {'backend_path': '/usr/lib/libQnnHtp.so'}),
             'CPUExecutionProvider']
            if config.use_qnn else ['CPUExecutionProvider']
        )
        session = ort.InferenceSession(config.model_path, providers=providers)
    except Exception as e:
        logger.error(f"[2/3] ONNX model load failed: {e}")
        return []

    input_name = session.get_inputs()[0].name
    inp_shape  = session.get_inputs()[0].shape
    iw = inp_shape[3] if isinstance(inp_shape[3], int) else config.input_size[0]
    ih = inp_shape[2] if isinstance(inp_shape[2], int) else config.input_size[1]
    logger.info(f"[2/3] Model loaded. Input: {iw}x{ih}")

    # 3. Preprocess → infer → postprocess
    img = cv2.resize(frame, (iw, ih))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    img = np.transpose(img, (2, 0, 1))[None]  # NCHW uint8

    logger.info("[3/3] Running inference...")
    try:
        outputs = session.run(None, {input_name: img})
    except Exception as e:
        logger.error(f"[3/3] Inference failed: {e}")
        return []

    detections = _postprocess(outputs, frame.shape, 0, iw, ih, config)
    logger.info(f"[3/3] {len(detections)} detection(s):")
    for d in detections:
        logger.info(f"  {d.class_name:<15} conf={d.confidence:.3f} bbox={d.bbox}")

    if detections and config.save_annotated_frames:
        annotated = _draw_detections(frame, detections)
        out_path = f"{config.output_dir}/{timestamp}_annotated.jpg"
        cv2.imwrite(out_path, annotated)
        logger.info(f"Annotated frame saved -> {out_path}")
    elif not detections:
        logger.info("No detections above threshold.")

    return detections
