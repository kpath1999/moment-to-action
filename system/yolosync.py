#!/usr/bin/env python3
"""
Temporal alignment helpers for YAMNet, Whisper, and YOLO outputs.

Provides:
- simple YOLO runner to emit `VisualEvent` objects
- dataclasses for audio, speech, vision events
- alignment + text rendering utilities consumed by the LLM stage
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import cv2
import onnxruntime as ort

import fight_module

# -------------------------- DATA CLASSES ---------------------------

@dataclass
class AudioEvent:
	start: float
	end: float
	label: str
	confidence: float


@dataclass
class AudioSpeech:
	start: float
	end: float
	label: str
	confidence: float


@dataclass
class VisualEvent:
	time: float
	objects: List[str]
	boxes: List[List[float]]
	confidences: List[float]


@dataclass
class FusedMoment:
	time: float
	audio: List[AudioEvent]
	speech: List[AudioSpeech]
	vision: List[VisualEvent]


# -------------------------- YOLO HELPERS ---------------------------

_COCO_CLASSES = [
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

_BOX_ZERO_POINT = 36
_BOX_SCALE      = 3.26531720161438
_SCORE_SCALE    = 0.00390625   # 1/256


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


# -------------------------- YOLO BB INFERENCE ---------------------------

def run_yolo_tracking(video_path: str, model_path: str = "yolov8n.onnx", display: bool = False, num_frames: int = 10) -> List[VisualEvent]:
	"""
	Run YOLO detection on evenly-sampled frames from a video.

	Args:
		video_path: Path to video file
		model_path: Path to ONNX YOLO model weights
		display: Whether to display annotated frames
		num_frames: Number of frames to sample evenly across the video (default: 10)

	Returns:
		List of VisualEvent objects with timestamps
	"""
	session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
	inp = session.get_inputs()[0]
	is_float_input = "float" in str(inp.type).lower()
	iw = inp.shape[3] if isinstance(inp.shape[3], int) else 640
	ih = inp.shape[2] if isinstance(inp.shape[2], int) else 640

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		return []

	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	if total_frames == 0:
		cap.release()
		return []

	# Calculate which frame indices to sample (evenly distributed)
	frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
	events: List[VisualEvent] = []

	for frame_idx in frame_indices:
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
		success, frame = cap.read()
		if not success:
			continue

		timestamp = frame_idx / fps

		# Preprocess
		img = cv2.resize(frame, (iw, ih))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = np.transpose(img, (2, 0, 1))[None]
		if is_float_input:
			img = img.astype(np.float32) / 255.0
		else:
			img = img.astype(np.uint8)

		outputs = session.run(None, {inp.name: img})

		# Standard YOLOv8 ONNX output shape is [1, 84, 8400]
		# 84 = 4 (box coords) + 80 (class scores)
		out = outputs[0][0]  # shape: [84, 8400]
		out = out.T          # shape: [8400, 84]

		boxes_f = out[:, :4]
		scores_raw = out[:, 4:]
		
		# Get max score and corresponding class for each box
		scores_f = np.max(scores_raw, axis=1)
		classes_f = np.argmax(scores_raw, axis=1)

		# Convert cx, cy, w, h to x1, y1, x2, y2
		cx, cy, w, h = boxes_f[:, 0], boxes_f[:, 1], boxes_f[:, 2], boxes_f[:, 3]
		x1 = cx - w / 2
		y1 = cy - h / 2
		x2 = cx + w / 2
		y2 = cy + h / 2
		boxes_f = np.stack([x1, y1, x2, y2], axis=1)

		mask = scores_f >= 0.5
		boxes_f, scores_f, classes_f = boxes_f[mask], scores_f[mask], classes_f[mask]

		if len(boxes_f) == 0:
			continue

		keep = _nms(boxes_f, scores_f, 0.45)
		boxes_f, scores_f, classes_f = boxes_f[keep], scores_f[keep], classes_f[keep]

		# Scale boxes back to original frame coordinates
		oh, ow = frame.shape[:2]
		sx, sy = ow / iw, oh / ih
		scaled_boxes = [
			[int(b[0]*sx), int(b[1]*sy), int(b[2]*sx), int(b[3]*sy)]
			for b in boxes_f
		]
		labels = [_COCO_CLASSES[int(c)] if int(c) < len(_COCO_CLASSES) else str(int(c)) for c in classes_f]
		events.append(VisualEvent(time=timestamp, objects=labels, boxes=scaled_boxes, confidences=scores_f.tolist()))

		if display:
			for (x1, y1, x2, y2), label in zip(scaled_boxes, labels):
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
				cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
			cv2.imshow("YOLO Detection", frame)
			if cv2.waitKey(500) & 0xFF == ord("q"):
				break

	cap.release()
	if display:
		cv2.destroyAllWindows()

	return events


# -------------------------- YOLO POSE DETECTION (TRIGGER) ---------------------------

def run_pose_detection(
    video_path: str,
    fight_model_path: str,
    yolo_model_path: str,
    likelihood_threshold: float = 0.30,
    min_trigger_frames: int = 3,
) -> tuple:
    """
    Run YOLO pose-based fight detection on a video as a lightweight first-stage trigger.

    Iterates over every frame produced by YoloPoseEstimation and computes a heuristic
    fight_likelihood score per frame.  Returns as soon as the pipeline is complete.

    Args:
        video_path: Path to the input video file.
        fight_model_path: Path to the FightDetector .pth weights.
        yolo_model_path: Path to the YOLOv8-pose .pt weights.
        likelihood_threshold: Per-frame fight_likelihood value that counts as a "hit"
            (default 0.30).
        min_trigger_frames: Number of frames that must meet the threshold before the
            function reports a trigger (default 3).

    Returns:
        (triggered, high_likelihood_count, frame_results)
          triggered             – True when >= min_trigger_frames frames exceeded the
                                  threshold.
          high_likelihood_count – Total number of frames that met the threshold.
          frame_results         – List of per-frame dicts with detection metadata.
    """
    fdet = fight_module.FightDetector(fight_model_path)
    yolo = fight_module.YoloPoseEstimation(yolo_model_path)

    frame_idx = 0
    high_likelihood_count = 0
    frame_results = []

    for result in yolo.estimate(video_path):
        frame_idx += 1

        people_count = 0
        interaction_boxes = []
        frame_fight_votes = 0
        max_iou = 0.0

        try:
            boxes = result.boxes.xyxy.tolist()
            xyn = result.keypoints.xyn.tolist()
            confs = result.keypoints.conf
            ids = result.boxes.id

            people_count = len(boxes)
            confs = [] if confs is None else confs.tolist()
            ids = [] if ids is None else [str(int(ID)) for ID in ids]

            interaction_boxes = fight_module.get_interaction_box(boxes)

            ious = fight_module.calculate_all_ious(boxes)
            if ious:
                max_iou = float(max(ious))

            for inter_box in interaction_boxes:
                both_fighting = []
                for conf, xyn_person, box, identity in zip(confs, xyn, boxes, ids):
                    cx = (box[2] + box[0]) / 2
                    cy = (box[3] + box[1]) / 2
                    if inter_box[0] <= cx <= inter_box[2] and inter_box[1] <= cy <= inter_box[3]:
                        is_person_fighting = fdet.detect(conf, xyn_person)
                        both_fighting.append(is_person_fighting)
                if both_fighting and all(both_fighting):
                    frame_fight_votes += 1

        except (TypeError, IndexError):
            pass

        overlap_score = min(1.0, max_iou / 0.5) if max_iou > 0 else 0.0
        vote_score = 1.0 if frame_fight_votes > 0 else 0.0
        interaction_score = 1.0 if len(interaction_boxes) > 0 else 0.0
        fight_likelihood = 0.5 * vote_score + 0.35 * overlap_score + 0.15 * interaction_score

        frame_result = {
            "frame": frame_idx,
            "people_count": people_count,
            "interaction_boxes": len(interaction_boxes),
            "max_iou": round(max_iou, 4),
            "fight_votes": frame_fight_votes,
            "fight_likelihood": round(fight_likelihood, 4),
        }
        frame_results.append(frame_result)

        if fight_likelihood >= likelihood_threshold:
            high_likelihood_count += 1
            print(
                f"[PoseDetect Frame {frame_idx}] people={people_count} "
                f"interaction_boxes={len(interaction_boxes)} "
                f"max_iou={max_iou:.3f} "
                f"fight_votes={frame_fight_votes} "
                f"fight_likelihood={fight_likelihood:.2f} "
                f"[threshold_hits={high_likelihood_count}/{min_trigger_frames}]"
            )

            if high_likelihood_count >= min_trigger_frames:
                print(f"Trigger threshold reached ({high_likelihood_count} frames). Stopping pose detection early.")
                break

    triggered = high_likelihood_count >= min_trigger_frames
    return triggered, high_likelihood_count, frame_results


# -------------------------- ALIGNMENT ---------------------------

def align_events(
		audio_events: List[AudioEvent],
		speech_events: List[AudioSpeech],
		visual_events: List[VisualEvent],
		window: float = 0.5,
		overlap: float = 0.25
) -> List[FusedMoment]:
	"""
	Bin events into overlapping temporal windows for LLM consumption.
	"""
	step = window - overlap

	max_time = max(
		max([a.end for a in audio_events], default=0.0),
		max([s.end for s in speech_events], default=0.0),
		max([v.time for v in visual_events], default=0.0)
	)

	timeline: List[FusedMoment] = []
	t = 0.0
	while t <= max_time:
		a_slice = [a for a in audio_events if not (a.end < t or a.start > t + window)]
		s_slice = [s for s in speech_events if not (s.end < t or s.start > t + window)]
		v_slice = [v for v in visual_events if t <= v.time <= t + window]

		timeline.append(FusedMoment(time=t, audio=a_slice, speech=s_slice, vision=v_slice))
		t += step

	return timeline


def timeline_to_text(timeline: List[FusedMoment]) -> str:
	"""
	Render fused moments into a compact textual summary for the LLM prompt.
	"""
	lines: List[str] = []

	for m in timeline:
		if not (m.audio or m.vision or m.speech):
			continue

		line = f"Time {m.time:.2f}s\n"

		if m.vision:
			objs = sorted({obj for v in m.vision for obj in v.objects})
			line += f"  Visual objects: {', '.join(objs)}\n"

		if m.audio:
			sounds = sorted({a.label for a in m.audio})
			line += f"  Sounds: {', '.join(sounds)}\n"

		if m.speech:
			transcript = " ".join([s.label for s in m.speech])
			line += f"  Speech: \"{transcript}\"\n"

		lines.append(line)

	return "\n".join(lines)

