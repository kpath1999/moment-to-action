#!/usr/bin/env python3
"""
Audio helpers for YAMNet classification and (placeholder) speech transcription.

This module focuses on taking raw audio segments and turning them into
`AudioEvent` and `AudioSpeech` instances that can be aligned with vision events.

Uses the quantized YAMNet TFLite model (yamnet_quantized.tflite) via ai_edge_litert
or tflite_runtime. Run helpers/get_yamnet_tflite.py to download the model file.
"""

import csv
import os
import threading
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

try:
	from ai_edge_litert.interpreter import Interpreter
except ImportError:
	from tflite_runtime.interpreter import Interpreter

from yolosync import AudioEvent, AudioSpeech


# -------------------------- YAMNET HELPERS ---------------------------

_YAMNET_INTERPRETER = None
_YAMNET_CLASS_NAMES = None
_WHISPER_MODEL = None
_WHISPER_LOCK = threading.Lock()
_WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "small")
_USE_GROQ_TRANSCRIPTION = os.environ.get("USE_GROQ_TRANSCRIPTION", "true").lower() in ("true", "1", "yes")

# Resolve model and label paths relative to this file
_HERE = Path(__file__).parent
_MODEL_PATH = str(_HERE / "yamnet_quantized.tflite")
_LABELS_PATH = str(_HERE / "yamnet_class_map.csv")


def _load_yamnet_model():
	"""Load the quantized YAMNet TFLite model and class labels once, then cache."""
	global _YAMNET_INTERPRETER, _YAMNET_CLASS_NAMES
	if _YAMNET_INTERPRETER is None:
		if not os.path.exists(_MODEL_PATH):
			raise FileNotFoundError(
				f"YAMNet TFLite model not found at {_MODEL_PATH}. "
				"Run helpers/get_yamnet_tflite.py to download it."
			)
		_YAMNET_INTERPRETER = Interpreter(model_path=_MODEL_PATH)
		_YAMNET_INTERPRETER.allocate_tensors()
		_YAMNET_CLASS_NAMES = _load_class_names(_LABELS_PATH)
	return _YAMNET_INTERPRETER, _YAMNET_CLASS_NAMES


def _load_class_names(csv_path: str) -> List[str]:
	"""Load YAMNet class display names from the class map CSV file."""
	class_names = []
	with open(csv_path, "r") as f:
		reader = csv.DictReader(f)
		for row in reader:
			class_names.append(row["display_name"])
	return class_names


def generate_audio_label(segment) -> Tuple[str, float]:
	"""Run YAMNet TFLite on one waveform segment and return top label and confidence."""
	interpreter, class_names = _load_yamnet_model()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	# Convert TF tensor or list to numpy float32 array
	if hasattr(segment, "numpy"):
		waveform = segment.numpy().astype(np.float32)
	else:
		waveform = np.array(segment, dtype=np.float32)

	# Quantize input if the model expects int8/uint8
	dtype = input_details[0]["dtype"]
	if dtype in (np.int8, np.uint8):
		scale, zero_point = input_details[0]["quantization"]
		waveform = (waveform / scale + zero_point).astype(dtype)

	interpreter.set_tensor(input_details[0]["index"], waveform)
	interpreter.invoke()

	scores = interpreter.get_tensor(output_details[0]["index"])

	# Dequantize output if needed
	out_dtype = output_details[0]["dtype"]
	if out_dtype in (np.int8, np.uint8):
		scale, zero_point = output_details[0]["quantization"]
		scores = (scores.astype(np.float32) - zero_point) * scale

	# scores shape is either [classes] or [time_frames, classes] — average over time
	scores = scores.squeeze()
	if scores.ndim == 2:
		scores = scores.mean(axis=0)

	top_idx = int(scores.argmax())
	return class_names[top_idx], float(scores[top_idx])


def analyze_audio_segments(segments: Iterable[Tuple[float, float, np.ndarray]]) -> List[AudioEvent]:
	"""
	Convert (start, end, waveform) tuples into AudioEvents using YAMNet.
	"""
	events: List[AudioEvent] = []
	for start, end, waveform in segments:
		label, confidence = generate_audio_label(waveform)
		events.append(AudioEvent(start=start, end=end, label=label, confidence=confidence))
	return events


def _load_whisper_model(model_name: str = _WHISPER_MODEL_NAME, device: str = "cpu"):
	"""Load local Whisper model once, then cache it."""
	global _WHISPER_MODEL
	with _WHISPER_LOCK:
		if _WHISPER_MODEL is None:
			import whisper
			_WHISPER_MODEL = whisper.load_model(model_name, device=device)
	return _WHISPER_MODEL


def preload_whisper(model_name: str = _WHISPER_MODEL_NAME, device: str = "cpu") -> None:
	"""
	Eagerly load and cache the Whisper model.

	Call this once at application startup (before the pipeline runs) so that
	the first call to transcribe_speech() does not pay the model-loading cost.
	"""
	_load_whisper_model(model_name=model_name, device=device)


# -------------------------- SPEECH TRANSCRIPTION ---------------------------

def transcribe_speech(
	audio_events: List[AudioEvent],
	segments: Iterable[Tuple[float, float, np.ndarray]],
	audio_file_path: str = None
) -> List[AudioSpeech]:
	"""
	Transcribe speech if YAMNet detected speech/voice.

	Primary method is controlled by USE_GROQ_TRANSCRIPTION env var (default: true).
	Set USE_GROQ_TRANSCRIPTION=false to use local Whisper as primary.
	"""
	# Check if speech was detected by YAMNet
	speech_detected = any(
		"speech" in event.label.lower() or "voice" in event.label.lower()
		for event in audio_events
	)
	
	if not speech_detected:
		return []

	segments = list(segments)

	# Route to primary transcription method based on flag
	if _USE_GROQ_TRANSCRIPTION:
		return _transcribe_groq_primary(segments, audio_file_path)
	else:
		return _transcribe_whisper_primary(segments, audio_file_path)


def _transcribe_groq_primary(
	segments: List[Tuple[float, float, np.ndarray]],
	audio_file_path: str = None
) -> List[AudioSpeech]:
	"""Groq as primary, local Whisper as fallback."""
	# 1) Try Groq first
	if audio_file_path:
		try:
			from groq import Groq

			client = Groq()

			with open(audio_file_path, "rb") as audio_file:
				transcription = client.audio.transcriptions.create(
					model="whisper-large-v3-turbo",
					file=audio_file,
					response_format="verbose_json",
					timestamp_granularities=["segment"]
				)

			speech_events: List[AudioSpeech] = []

			if hasattr(transcription, "segments") and transcription.segments:
				for segment in transcription.segments:
					speech_events.append(AudioSpeech(
						start=segment["start"],
						end=segment["end"],
						label=segment["text"],
						confidence=1.0
					))
			else:
				speech_events.append(AudioSpeech(
					start=0.0,
					end=0.0,
					label=transcription.text,
					confidence=1.0
				))

			return speech_events

		except Exception as e:
			print(f"Warning: Groq transcription failed, trying local Whisper fallback: {e}")

	# 2) Fallback to local Whisper
	try:
		if segments:
			full_audio = np.concatenate([np.asarray(waveform, dtype=np.float32) for _, _, waveform in segments])
			if np.abs(full_audio).max() > 0:
				full_audio = full_audio / np.abs(full_audio).max()

			whisper_model = _load_whisper_model(model_name=_WHISPER_MODEL_NAME, device="cpu")
			result = whisper_model.transcribe(full_audio, language="en", fp16=False)
			text = (result.get("text") or "").strip()

			if text:
				start = float(segments[0][0])
				end = float(segments[-1][1])
				return [AudioSpeech(start=start, end=end, label=text, confidence=1.0)]
	except Exception as e:
		print(f"Warning: Speech transcription failed (Groq + local Whisper fallback): {e}")

	return []


def _transcribe_whisper_primary(
	segments: List[Tuple[float, float, np.ndarray]],
	audio_file_path: str = None
) -> List[AudioSpeech]:
	"""Local Whisper as primary, Groq as fallback."""
	# 1) Try local Whisper first
	try:
		if segments:
			full_audio = np.concatenate([np.asarray(waveform, dtype=np.float32) for _, _, waveform in segments])
			if np.abs(full_audio).max() > 0:
				full_audio = full_audio / np.abs(full_audio).max()

			whisper_model = _load_whisper_model(model_name=_WHISPER_MODEL_NAME, device="cpu")
			result = whisper_model.transcribe(full_audio, language="en", fp16=False)
			text = (result.get("text") or "").strip()

			if text:
				start = float(segments[0][0])
				end = float(segments[-1][1])
				return [AudioSpeech(start=start, end=end, label=text, confidence=1.0)]
	except Exception as e:
		print(f"Warning: Local Whisper transcription failed, trying Groq fallback: {e}")

	# 2) Fallback to Groq if audio file path exists
	if not audio_file_path:
		return []

	try:
		from groq import Groq

		client = Groq()

		with open(audio_file_path, "rb") as audio_file:
			transcription = client.audio.transcriptions.create(
				model="whisper-large-v3-turbo",
				file=audio_file,
				response_format="verbose_json",
				timestamp_granularities=["segment"]
			)

		speech_events: List[AudioSpeech] = []

		if hasattr(transcription, "segments") and transcription.segments:
			for segment in transcription.segments:
				speech_events.append(AudioSpeech(
					start=segment["start"],
					end=segment["end"],
					label=segment["text"],
					confidence=1.0
				))
		else:
			speech_events.append(AudioSpeech(
				start=0.0,
				end=0.0,
				label=transcription.text,
				confidence=1.0
			))

		return speech_events

	except Exception as e:
		print(f"Warning: Speech transcription failed (local Whisper + Groq fallback): {e}")

	return []