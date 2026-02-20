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

import cv2
from ultralytics import YOLO

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


# -------------------------- YOLO INFERENCE ---------------------------

def run_yolo_tracking(video_path: str, model_path: str = "yolov8n.pt", display: bool = False, num_frames: int = 10) -> List[VisualEvent]:
	"""
	Run YOLO detection on evenly-sampled frames from a video.
	
	Args:
		video_path: Path to video file
		model_path: Path to YOLO model weights
		display: Whether to display annotated frames
		num_frames: Number of frames to sample evenly across the video (default: 10)
	
	Returns:
		List of VisualEvent objects with timestamps
	"""
	model = YOLO(model_path)
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

		results = model(frame)  # Use model() instead of model.track() for single-frame inference
		if not results or len(results) == 0:
			continue

		boxes = results[0].boxes.xyxy.cpu().tolist() if results[0].boxes is not None else []
		confs = results[0].boxes.conf.cpu().tolist() if results[0].boxes is not None else []
		classes = results[0].boxes.cls.cpu().tolist() if results[0].boxes is not None else []
		names = results[0].names

		labels = [names[int(c)] for c in classes]
		events.append(VisualEvent(time=timestamp, objects=labels, boxes=boxes, confidences=confs))

		if display:
			annotated_frame = results[0].plot()
			cv2.imshow("YOLO Detection", annotated_frame)
			if cv2.waitKey(500) & 0xFF == ord("q"):  # Show each frame for 500ms
				break

	cap.release()
	if display:
		cv2.destroyAllWindows()

	return events


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
