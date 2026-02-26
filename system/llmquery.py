#!/usr/bin/env python3
"""
LLM-facing helpers: synthesise all upstream evidence into MobileCLIP queries.

By the time the LLM is called, YOLO pose detection AND MoviNet have already
flagged the clip as suspicious.  The LLM's job is to fuse every upstream
signal into concrete, *visually-grounded* text queries that MobileCLIP can
use to retrieve the most relevant frames.

Input : pose results, MoviNet segments, YAMNet audio events, speech
        transcripts, YOLO bounding-box detections
Output: JSON with a short summary and MobileCLIP queries per moment
"""

import json
from typing import Any, Dict, Iterable, List, Union

from groq import Groq

from yolosync import AudioEvent, AudioSpeech, VisualEvent, align_events, timeline_to_text


# ───────────────────── upstream evidence → text ─────────────────────

def _format_pose_summary(pose_frame_results: List[Dict[str, Any]]) -> str:
	"""Condense per-frame pose results into a compact block."""
	if not pose_frame_results:
		return "No pose data available."

	total = len(pose_frame_results)
	hits = [f for f in pose_frame_results if f.get("fight_likelihood", 0) >= 0.30]
	hit_count = len(hits)
	max_likelihood = max(f.get("fight_likelihood", 0) for f in pose_frame_results)
	max_people = max(f.get("people_count", 0) for f in pose_frame_results)
	avg_iou = sum(f.get("max_iou", 0) for f in hits) / hit_count if hit_count else 0.0

	lines = [
		f"Frames analysed: {total}",
		f"Frames with fight_likelihood>=0.30: {hit_count}/{total}",
		f"Peak fight_likelihood: {max_likelihood:.2f}",
		f"Max people in a frame: {max_people}",
		f"Avg max_iou in hit frames: {avg_iou:.3f}",
	]
	# Include the top-5 highest-likelihood frames for detail
	top5 = sorted(pose_frame_results, key=lambda f: f.get("fight_likelihood", 0), reverse=True)[:5]
	for f in top5:
		lines.append(
			f"  Frame {f['frame']}: people={f['people_count']} "
			f"iou={f['max_iou']:.3f} votes={f['fight_votes']} "
			f"likelihood={f['fight_likelihood']:.2f}"
		)
	return "\n".join(lines)


def _format_movinet_summary(segment_predictions: List[Dict[str, Any]]) -> str:
	"""Condense MoviNet segment predictions into a compact block."""
	if not segment_predictions:
		return "No MoviNet data available."

	lines = [f"Segments: {len(segment_predictions)}"]
	for seg in segment_predictions:
		label = seg.get("label", "?")
		fp = seg.get("fight_prob", 0.0)
		lines.append(
			f"  [{seg.get('start_time', 0):.1f}s–{seg.get('end_time', 0):.1f}s] "
			f"{label} (fight_prob={fp:.3f})"
		)
	return "\n".join(lines)


def _build_evidence_text(
	pose_frame_results: List[Dict[str, Any]],
	movinet_segments: List[Dict[str, Any]],
	timeline_text: str,
) -> str:
	"""Merge all upstream evidence into a single text block for the LLM."""
	sections = []

	sections.append("=== YOLO POSE DETECTION ===")
	sections.append(_format_pose_summary(pose_frame_results))

	sections.append("\n=== MOVINET VIDEO-LEVEL CLASSIFICATION ===")
	sections.append(_format_movinet_summary(movinet_segments))

	sections.append("\n=== TEMPORAL AUDIO / SPEECH / OBJECT DETECTIONS ===")
	sections.append(timeline_text if timeline_text.strip() else "No detections.")

	return "\n".join(sections)


# ───────────────────── LLM call ─────────────────────

_SYSTEM_PROMPT = """\
You are a violence-analysis assistant that turns multi-modal evidence into \
visual search queries for MobileCLIP (a vision-language model that matches \
text to images).

CONTEXT: Two upstream detectors — YOLO pose-based fight detection and MoviNet \
video classification — have ALREADY flagged this clip as likely containing \
physical violence.  Your job is NOT to re-estimate the probability; instead, \
synthesise the evidence below and produce targeted visual search queries that \
will help MobileCLIP retrieve the most relevant frames.

RULES FOR mobileclip_queries:
1. Each query must describe something VISIBLE in a single still frame — \
body posture, spatial relationships between people, objects, actions.
2. Do NOT reference audio, speech, emotions, intentions, or off-screen events.
3. Be specific and concrete (e.g. "two people grappling on the ground", \
"person throwing a punch at another person", \
"person pinned against a wall by another person").
4. Vary the queries to cover different visual aspects: body positions, \
proximity, weapons/objects, environment.
5. Produce EXACTLY 2 or 3 queries total across the entire clip — not per moment.

OUTPUT STRICT JSON ONLY — no text outside the JSON object.

{
  "summary": "<one-paragraph description of what is happening in the clip>",
  "mobileclip_queries": ["<visual query 1>", "<visual query 2>", "<optional visual query 3>"],
  "moments_of_interest": [
    {
      "start": <float seconds>,
      "end": <float seconds>,
      "description": "<what is visually happening in this interval>"
    }
  ]
}
"""


def query_llm(evidence_text: str, question: str) -> dict:
	"""Query Groq Llama model with upstream evidence and a strict JSON schema."""
	client = Groq()

	user_prompt = f"""\
UPSTREAM EVIDENCE
{evidence_text}

QUESTION: {question}
"""

	completion = client.chat.completions.create(
		model="llama-3.3-70b-versatile",
		temperature=0.1,
		messages=[
			{"role": "system", "content": _SYSTEM_PROMPT},
			{"role": "user", "content": user_prompt},
		],
		response_format={"type": "json_object"},
	)

	return json.loads(completion.choices[0].message.content)


# ───────────────────── coercion helpers ─────────────────────

def _coerce_events(events: Iterable[Union[AudioEvent, AudioSpeech, VisualEvent]], cls):
	coerced = []
	for e in events:
		if isinstance(e, cls):
			coerced.append(e)
		else:
			coerced.append(cls(**e))
	return coerced


# ───────────────────── public entry point ─────────────────────

def analyze_video_with_llm(
		yamnet_predictions: Iterable[Union[AudioEvent, dict]],
		whisper_predictions: Iterable[Union[AudioSpeech, dict]],
		yolo_predictions: Iterable[Union[VisualEvent, dict]],
		pose_frame_results: List[Dict[str, Any]] = None,
		movinet_segments: List[Dict[str, Any]] = None,
		question: str = "Describe the violent actions visible in this clip.",
) -> dict:
	"""
	Fuse all upstream evidence and ask the LLM for MobileCLIP-ready queries.
	"""
	audio_events = _coerce_events(yamnet_predictions, AudioEvent)
	speech_events = _coerce_events(whisper_predictions, AudioSpeech)
	visual_events = _coerce_events(yolo_predictions, VisualEvent)

	timeline = align_events(audio_events, speech_events, visual_events)
	timeline_text = timeline_to_text(timeline)

	evidence_text = _build_evidence_text(
		pose_frame_results or [],
		movinet_segments or [],
		timeline_text,
	)

	return query_llm(evidence_text, question)
