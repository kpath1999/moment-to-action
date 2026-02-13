#!/usr/bin/env python3
"""
Audio helpers for YAMNet classification and (placeholder) speech transcription.

This module focuses on taking raw audio segments and turning them into
`AudioEvent` and `AudioSpeech` instances that can be aligned with vision events.
"""

import csv
import io
from typing import Iterable, List, Tuple

import tensorflow as tf
import tensorflow_hub as hub

from yolosync import AudioEvent, AudioSpeech


# -------------------------- YAMNET HELPERS ---------------------------

def _load_yamnet_model():
	"""Load YAMNet from TF Hub once and reuse it."""
	return hub.load("https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1")


def class_names_from_csv(class_map_csv_text: str) -> List[str]:
	"""Return list of class names corresponding to score vector."""
	class_map_csv = io.StringIO(class_map_csv_text)
	class_names = [display_name for (_idx, _mid, display_name) in csv.reader(class_map_csv)]
	return class_names[1:]  # skip header


def generate_audio_label(segment) -> Tuple[str, float]:
	"""Run YAMNet on one waveform segment and return top label and confidence."""
	model = _load_yamnet_model()

	# segment is expected to be a 1-D float32 waveform tensor
	scores, _embeddings, _log_mel_spectrogram = model(segment)
	scores.shape.assert_is_compatible_with([None, 521])

	class_map_path = model.class_map_path().numpy()
	class_names = class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode("utf-8"))

	avg_scores = scores.numpy().mean(axis=0)
	top_idx = int(avg_scores.argmax())
	return class_names[top_idx], float(avg_scores[top_idx])


def analyze_audio_segments(segments: Iterable[Tuple[float, float, tf.Tensor]]) -> List[AudioEvent]:
	"""
	Convert (start, end, waveform) tuples into AudioEvents using YAMNet.
	"""
	events: List[AudioEvent] = []
	for start, end, waveform in segments:
		label, confidence = generate_audio_label(waveform)
		events.append(AudioEvent(start=start, end=end, label=label, confidence=confidence))
	return events


# -------------------------- SPEECH TRANSCRIPTION ---------------------------

def transcribe_speech(
	audio_events: List[AudioEvent],
	segments: Iterable[Tuple[float, float, tf.Tensor]],
	audio_file_path: str = None
) -> List[AudioSpeech]:
	"""
	Transcribe speech using OpenAI Whisper if YAMNet detected 'Speech' class.
	
	Args:
		audio_events: YAMNet results to check if speech was detected
		segments: Audio segments with timing info (not used if audio_file_path provided)
		audio_file_path: Optional path to audio file for transcription
	
	Returns:
		List of AudioSpeech events with transcribed text
	"""
	# Check if speech was detected by YAMNet
	speech_detected = any(
		"speech" in event.label.lower() or "voice" in event.label.lower()
		for event in audio_events
	)
	
	if not speech_detected:
		return []
	
	# If no audio file path provided, cannot transcribe
	if not audio_file_path:
		return []
	
	try:
		from openai import OpenAI
		
		client = OpenAI()
		
		with open(audio_file_path, "rb") as audio_file:
			transcription = client.audio.transcriptions.create(
				model="whisper-1",
				file=audio_file,
				response_format="verbose_json",
				timestamp_granularities=["segment"]
			)
		
		# Convert OpenAI response to AudioSpeech events
		speech_events: List[AudioSpeech] = []
		
		if hasattr(transcription, 'segments'):
			for segment in transcription.segments:
				speech_events.append(AudioSpeech(
					start=segment['start'],
					end=segment['end'],
					label=segment['text'],
					confidence=1.0  # OpenAI doesn't provide confidence scores
				))
		else:
			# Fallback: single transcription without timing
			speech_events.append(AudioSpeech(
				start=0.0,
				end=0.0,
				label=transcription.text,
				confidence=1.0
			))
		
		return speech_events
		
	except Exception as e:
		# Silently fail if OpenAI not configured or other errors
		print(f"Warning: Speech transcription failed: {e}")
		return []