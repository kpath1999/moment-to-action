"""
Segment-based fight detection using MoviNet.

The video is split into fixed-duration segments (default 3 s). MoviNet is run
independently on each segment with a fresh recurrent state. If Fight probability
exceeds the threshold for a configurable number of consecutive segments (default 3),
the downstream pipeline is triggered.
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


@dataclass
class SegmentPrediction:
	"""Prediction for a single time segment."""
	segment_index: int
	start_time: float   # seconds
	end_time: float     # seconds
	label: str
	fight_prob: float
	no_fight_prob: float


# Backward-compatible alias
FramePrediction = SegmentPrediction


def get_top_k(probs: tf.Tensor, k: int = 2, label_map: List[str] = MOVINET_CLASSES):
	"""Outputs the top k model labels and probabilities on the given video."""
	top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
	top_indices = [int(x) for x in top_predictions.numpy().tolist()]
	top_labels = [label_map[index] for index in top_indices]
	top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
	return tuple(zip(top_labels, top_probs))


def frames_from_video_segment(
		video_path: str,
		start_frame: int,
		end_frame: int,
		n_frames: int,
		output_size=(172, 172),
) -> np.ndarray:
	"""
	Sample n_frames evenly from [start_frame, end_frame) and return as (n_frames, H, W, C).
	Frames are BGR→RGB converted and float32 normalised.
	"""
	src = cv2.VideoCapture(str(video_path))
	if not src.isOpened():
		raise ValueError(f"Unable to open video file: {video_path}")

	result = []
	indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int)
	for idx in indices:
		src.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
		ret, frame = src.read()
		if ret:
			result.append(format_frames(frame, output_size))
		elif result:
			result.append(np.zeros_like(result[0]))
		else:
			result.append(np.zeros((*output_size, 3), dtype=np.float32))

	src.release()
	# Stack and flip BGR → RGB
	return np.array(result)[..., [2, 1, 0]]


def stream_segment_inference(
		video_path: str,
		model_path: str | None = None,
		image_size=(172, 172),
		n_frames: int = 12,
		segment_fraction: float = 0.1,
		label_map: List[str] = MOVINET_CLASSES,
):
	"""
	Generator that yields per-segment Fight/No_Fight predictions.

	The video is divided into non-overlapping windows, each `segment_fraction`
	of the total frame count (default 0.1 = 10%). Segments are processed in order
	from the start of the video. MoviNet recurrent states are reset at the start
	of every segment so each segment is classified independently.

	Yields:
		SegmentPrediction for each segment.
	"""
	src = cv2.VideoCapture(str(video_path))
	if not src.isOpened():
		raise ValueError(f"Unable to open video file: {video_path}")
	fps = src.get(cv2.CAP_PROP_FPS) or 30.0
	total_frames = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
	src.release()

	segment_frames = max(1, int(total_frames * segment_fraction))
	runner, init_states = build_movinet_runner(model_path=model_path)

	segment_idx = 0
	for seg_start in range(0, total_frames, segment_frames):
		seg_end = min(seg_start + segment_frames, total_frames)
		if seg_end <= seg_start:
			break

		start_time = seg_start / fps
		end_time = seg_end / fps

		# Sample n_frames evenly across this segment
		frames = frames_from_video_segment(
			video_path, seg_start, seg_end, n_frames, output_size=image_size
		)
		video_tensor = tf.convert_to_tensor(frames, dtype=tf.float32)
		clips = tf.split(video_tensor[tf.newaxis], video_tensor.shape[0], axis=1)

		# Fresh state for every segment
		states = dict(init_states)
		logits = None
		for clip in clips:
			outputs = runner(**states, image=clip)
			logits = outputs.pop("logits")[0]
			states = outputs

		if logits is None:
			segment_idx += 1
			continue

		probs = tf.nn.softmax(logits).numpy()
		fight_prob = float(probs[label_map.index("Fight")])
		no_fight_prob = float(probs[label_map.index("No_Fight")])
		label = "Fight" if fight_prob >= no_fight_prob else "No_Fight"

		yield SegmentPrediction(
			segment_index=segment_idx,
			start_time=start_time,
			end_time=end_time,
			label=label,
			fight_prob=fight_prob,
			no_fight_prob=no_fight_prob,
		)
		segment_idx += 1


def detect_fight_trigger(
		video_path: str,
		threshold: float = 0.5,
		model_path: str | None = None,
		image_size=(172, 172),
		n_frames: int = 12,
		segment_fraction: float = 0.1,
		consecutive_segments: int = 3,
) -> Tuple[bool, List[SegmentPrediction], int | None]:
	"""
	Process video in fixed-fraction segments and detect sustained fighting.

	Each segment spans `segment_fraction` of the total video length (default
	0.1 = 10%). Segments are evaluated from the start of the video in order.
	A trigger fires when `consecutive_segments` back-to-back segments all have
	a Fight probability above `threshold` (default: 3 × 10% = 30% of clip).

	Args:
		video_path: Path to video file.
		threshold: Per-segment Fight probability threshold (default 0.5).
		model_path: Optional path to MoViNet TFLite model.
		image_size: Frame resize dimensions.
		n_frames: Frames sampled per segment for inference.
		segment_fraction: Each segment as a fraction of total video length (default 0.1).
		consecutive_segments: How many consecutive fight segments trigger the pipeline (default 3).

	Returns:
		Tuple of (triggered, all_predictions, trigger_segment_index).
		triggered is True when the consecutive threshold was met.
	"""
	predictions: List[SegmentPrediction] = []
	triggered = False
	trigger_segment = None
	consecutive_count = 0

	for pred in stream_segment_inference(
		video_path=video_path,
		model_path=model_path,
		image_size=image_size,
		n_frames=n_frames,
		segment_fraction=segment_fraction,
	):
		predictions.append(pred)

		if pred.fight_prob > threshold:
			consecutive_count += 1
			if consecutive_count >= consecutive_segments:
				triggered = True
				trigger_segment = pred.segment_index
				break  # Stop processing further segments
		else:
			consecutive_count = 0

	return triggered, predictions, trigger_segment

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