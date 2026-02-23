#!/usr/bin/env python3
"""
Temporal alignment helpers for YAMNet, Whisper, and YOLO outputs.

Provides:
- simple YOLO runner to emit `VisualEvent` objects
- dataclasses for audio, speech, vision events
- alignment + text rendering utilities consumed by the LLM stage
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import os
import random
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from ai_edge_litert.interpreter import Interpreter

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


# -------------------------- MOVINET INFERENCE ---------------------------

MOVINET_CLASSES = ["Fight", "No_Fight"]


def _default_movinet_model_path() -> str:
	base_dir = os.path.dirname(os.path.abspath(__file__))
	return os.path.join(base_dir, "movinet.tflite")


def build_movinet_runner(model_path: str | None = None) -> Tuple[Any, Dict[str, tf.Tensor]]:
	"""Create LiteRT signature runner and zero-initialized recurrent states."""
	resolved_model_path = model_path or _default_movinet_model_path()
	interpreter = Interpreter(model_path=resolved_model_path)
	runner = interpreter.get_signature_runner()

	init_states = {
		name: tf.zeros(details["shape"], dtype=details["dtype"])
		for name, details in runner.get_input_details().items()
	}
	if "image" in init_states:
		del init_states["image"]

	return runner, init_states

def frames_from_video_file(video_path: str, n_frames: int, output_size=(224, 224), frame_step: int = 15) -> np.ndarray:
	"""
		Creates frames from each video file present for each category.

		Args:
		video_path: File path to the video.
		n_frames: Number of frames to be created per video file.
		output_size: Pixel size of the output frame image.

		Return:
		An NumPy array of frames in the shape of (n_frames, height, width, channels).
	"""

	# Read each video frame by frame
	result = []
	src = cv2.VideoCapture(str(video_path))
	if not src.isOpened():
		raise ValueError(f"Unable to open video file: {video_path}")

	video_length = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
	
	need_length = 1 + (n_frames - 1) * frame_step

	if need_length > video_length:
		start = 0
	else:
		max_start = video_length - need_length
		start = random.randint(0, max_start + 1)

	src.set(cv2.CAP_PROP_POS_FRAMES, start)
	# ret is a boolean indicating whether read was successful, frame is the image itself
	ret, frame = src.read()
	if not ret:
		src.release()
		raise ValueError(f"Unable to read frames from video file: {video_path}")
	
	result.append(format_frames(frame, output_size))

	for _ in range(n_frames - 1):
		for _ in range(frame_step):
			ret, frame = src.read()
			if ret:
				frame = format_frames(frame, output_size)
				result.append(frame)
			else:
				result.append(np.zeros_like(result[0]))
	
	src.release()
	result = np.array(result)[..., [2, 1, 0]]

	return result

def format_frames(frame: np.ndarray, output_size) -> tf.Tensor:
	"""
		Pad and resize an image from a video.

		Args:
		frame: Image that needs to resized and padded.
		output_size: Pixel size of the output frame image.

		Return:
		Formatted frame with padding of specified output size.
	"""
	frame = tf.image.convert_image_dtype(frame, tf.float32)
	frame = tf.image.resize_with_pad(frame, *output_size)
	return frame

def video_to_tensor(video_path: str, image_size=(172, 172), n_frames: int = 12, frame_step: int = 15) -> tf.Tensor:
	"""Load and preprocess video frames into a [T, H, W, 3] float32 tensor."""
	frames = frames_from_video_file(
		video_path=video_path,
		n_frames=n_frames,
		output_size=image_size,
		frame_step=frame_step,
	)
	return tf.convert_to_tensor(frames, dtype=tf.float32)


def get_top_k(probs: tf.Tensor, k: int = 2, label_map: List[str] = MOVINET_CLASSES):
	"""Outputs the top k model labels and probabilities on the given video."""
	top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
	top_indices = [int(x) for x in top_predictions.numpy().tolist()]
	top_labels = [label_map[index] for index in top_indices]
	top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
	return tuple(zip(top_labels, top_probs))

def run_movinet_inference(
		video_path: str,
		model_path: str | None = None,
		image_size=(172, 172),
		n_frames: int = 12,
		frame_step: int = 15,
		top_k: int = 2,
		label_map: List[str] = MOVINET_CLASSES,
) -> Dict[str, Any]:
	"""Run LiteRT MoViNet inference on a single clip and return probabilities."""
	runner, init_states = build_movinet_runner(model_path=model_path)
	video = video_to_tensor(video_path, image_size=image_size, n_frames=n_frames, frame_step=frame_step)
	clips = tf.split(video[tf.newaxis], video.shape[0], axis=1)

	states = dict(init_states)
	logits = None
	for clip in clips:
		outputs = runner(**states, image=clip)
		logits = outputs.pop("logits")[0]
		states = outputs

	if logits is None:
		raise ValueError(f"No logits produced for video: {video_path}")

	probs = tf.nn.softmax(logits)
	top_predictions = get_top_k(probs, k=top_k, label_map=label_map)
	class_probabilities = {
		label: float(prob)
		for label, prob in zip(label_map, probs.numpy().tolist())
	}

	return {
		"top_k": [{"label": label, "probability": float(prob)} for label, prob in top_predictions],
		"class_probabilities": class_probabilities,
	}


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
