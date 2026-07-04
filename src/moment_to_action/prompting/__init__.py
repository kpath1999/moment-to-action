"""Prompt-building helpers shared by LLM/VLM stages and benchmark scripts.

Pure functions, no I/O: spatial context derivation from bounding boxes, chat
templates, and prompt/payload builders.
"""

from __future__ import annotations

from ._builder import DEFAULT_ANIMAL_LABELS, build_detection_prompt, build_payload
from ._grammar import YES_NO_GRAMMAR
from ._spatial import area, depth, frame_zone, iou, is_horizontal
from ._templates import BENCHMARK_SYSTEM, CHATML, PHI3

__all__ = [
    "BENCHMARK_SYSTEM",
    "CHATML",
    "DEFAULT_ANIMAL_LABELS",
    "PHI3",
    "YES_NO_GRAMMAR",
    "area",
    "build_detection_prompt",
    "build_payload",
    "depth",
    "frame_zone",
    "iou",
    "is_horizontal",
]
