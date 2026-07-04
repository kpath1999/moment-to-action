"""Prompt and request-payload builders shared by the LLM/VLM stages and benches."""

from __future__ import annotations

from typing import TYPE_CHECKING

from moment_to_action.prompting._spatial import (
    MIN_PAIR,
    OVERLAP_THRESH,
    frame_zone,
    is_horizontal,
)
from moment_to_action.prompting._spatial import depth as _depth
from moment_to_action.prompting._spatial import iou as _iou

if TYPE_CHECKING:
    from moment_to_action.models.image.detection._types import Detection

DEFAULT_ANIMAL_LABELS = frozenset({"dog", "cat", "bear", "wolf"})
"""Default label set considered "animal" for person-animal overlap context."""


def build_detection_prompt(
    detections: list[Detection],
    question: str,
    *,
    extra_lines: list[str] | None = None,
    animal_labels: frozenset[str] = DEFAULT_ANIMAL_LABELS,
) -> str:
    """Build a model prompt from detector output and a binary question.

    Spatial features (overlap, orientation, foreground/background) are derived
    from bounding box coordinates rather than assumed from free text — no
    language appears in the prompt that could not be computed from real
    detector output.

    Args:
        detections: Detections to describe, in detector output order.
        question: The binary question to append after the detection context.
        extra_lines: Optional extra context lines (e.g. an audio transcript)
            inserted after the detection/spatial context and before *question*.
        animal_labels: Labels considered "animal" for the person-animal overlap
            line. Defaults to :data:`DEFAULT_ANIMAL_LABELS`; callers with a
            broader label set (e.g. all COCO animal classes) can override it.

    Returns:
        Formatted prompt string ending with *question*.
    """
    lines: list[str] = [f"Task: {question}", ""]

    det_lines: list[str] = []
    for d in detections:
        zone = frame_zone(d.bbox)
        dep = _depth(d.bbox)
        parts = [f"{d.label} (conf {d.confidence:.2f}, {zone}, {dep}"]
        if d.label == "person" and is_horizontal(d.bbox):
            parts.append(", horizontal orientation")
        parts.append(")")
        det_lines.append("".join(parts))
    lines.append("Detections:\n" + "\n".join(f"  - {dl}" for dl in det_lines))

    persons = [d for d in detections if d.label == "person"]
    animals = [d for d in detections if d.label in animal_labels]

    if len(persons) >= MIN_PAIR:
        max_person_iou = max(
            _iou(persons[i].bbox, persons[j].bbox)
            for i in range(len(persons))
            for j in range(i + 1, len(persons))
        )
        overlap_desc = "overlapping" if max_person_iou > OVERLAP_THRESH else "non-overlapping"
        lines.append(f"Person bounding boxes: {overlap_desc} (max IoU={max_person_iou:.2f})")

    if persons and animals:
        max_pa_iou = max(_iou(p.bbox, a.bbox) for p in persons for a in animals)
        pa_desc = "overlapping with person" if max_pa_iou > OVERLAP_THRESH else "not overlapping"
        lines.append(f"Animal bounding box: {pa_desc} (max IoU with person={max_pa_iou:.2f})")

    if extra_lines:
        lines.extend(extra_lines)

    lines.append("")
    lines.append(question)
    return "\n".join(lines)


def build_payload(
    prompt: str,
    max_tokens: int,
    system_prompt: str,
    template: str | None,
) -> dict:
    """Build a llama.cpp ``/completion`` request dict for text-only inference.

    Works for both ``LlamaGGUFModel`` and ``LlamaVLModel`` (text-only mode),
    bypassing ``model._prepare()`` so a chat-template model can be driven
    without passing image tuples.

    If *template* is provided it must contain ``{system}`` and ``{user}``
    placeholders, substituted with *system_prompt* and *prompt* respectively
    (see :mod:`moment_to_action.prompting._templates`). When *template* is
    ``None`` the system prompt is prepended raw (system + newline + prompt).

    Args:
        prompt: User prompt text.
        max_tokens: Maximum tokens to generate.
        system_prompt: System message text.
        template: Optional format string with ``{system}``/``{user}`` placeholders.

    Returns:
        ``/completion`` request body dict.
    """
    if template is not None:
        full_prompt = template.format(system=system_prompt, user=prompt)
    else:
        full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt
    return {"prompt": full_prompt, "n_predict": max_tokens}
