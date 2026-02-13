#!/usr/bin/env python3
"""
LLM-facing helpers: turn fused timeline text into structured JSON.

Input: aligned audio, speech, and vision events
Output: JSON containing violence probability and MobileCLIP-friendly queries
"""

import json
from typing import Iterable, List, Union

from groq import Groq

from yolosync import AudioEvent, AudioSpeech, VisualEvent, align_events, timeline_to_text


def query_llm(timeline_text: str, question: str) -> dict:
	"""Query Groq Llama model with a strict JSON schema."""
	client = Groq()

	system_prompt = """
You analyze synchronized audio, speech, and object detections from a video.

Goal:
1) Infer what is happening
2) Estimate probability of physical violence
3) Produce MobileCLIP visual search queries describing aggressive actions

Output STRICT JSON ONLY following schema:

{
	violence_probability: float (0-1),
	summary: string,
	moments_of_interest: [
		{
			start: float,
			end: float,
			reason: string,
			mobileclip_queries: [string]
		}
	]
}

Rules:
- Queries must describe visible actions, not emotions
- No explanations outside JSON 
"""

	user_prompt = f"""
VIDEO OBSERVATIONS
{timeline_text}

QUESTION: {question}
"""

	completion = client.chat.completions.create(
		model="llama3-70b-8192",
		temperature=0.1,
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
		response_format={"type": "json_object"}
	)

	return json.loads(completion.choices[0].message.content)


def _coerce_events(events: Iterable[Union[AudioEvent, AudioSpeech, VisualEvent]], cls):
	coerced = []
	for e in events:
		if isinstance(e, cls):
			coerced.append(e)
		else:
			coerced.append(cls(**e))
	return coerced


def analyze_video_with_llm(
		yamnet_predictions: Iterable[Union[AudioEvent, dict]],
		whisper_predictions: Iterable[Union[AudioSpeech, dict]],
		yolo_predictions: Iterable[Union[VisualEvent, dict]],
		question: str = "Is violence likely?"
) -> dict:
	"""
	Align events, compress to text, and ask the LLM for structured reasoning.
	"""
	audio_events = _coerce_events(yamnet_predictions, AudioEvent)
	speech_events = _coerce_events(whisper_predictions, AudioSpeech)
	visual_events = _coerce_events(yolo_predictions, VisualEvent)

	timeline = align_events(audio_events, speech_events, visual_events)
	timeline_text = timeline_to_text(timeline)

	return query_llm(timeline_text, question)
