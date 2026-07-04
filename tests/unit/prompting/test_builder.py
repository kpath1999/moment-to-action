"""Unit tests for prompting._builder."""

from __future__ import annotations

import pytest

from moment_to_action.models.image.detection._types import BoundingBox, Detection
from moment_to_action.prompting import build_detection_prompt, build_payload
from moment_to_action.prompting._templates import CHATML


def _det(label: str, conf: float, x1: float, y1: float, x2: float, y2: float) -> Detection:
    """Shorthand Detection constructor for tests."""
    return Detection(label=label, confidence=conf, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))


@pytest.mark.unit
class TestBuildDetectionPrompt:
    """Tests for build_detection_prompt()."""

    def test_includes_question(self) -> None:
        """The question appears at the start and end of the prompt."""
        prompt = build_detection_prompt([], "Is this safe?")
        assert "Is this safe?" in prompt

    def test_includes_detection_lines(self) -> None:
        """Each detection produces a descriptive line."""
        dets = [_det("person", 0.9, 10, 10, 100, 400)]
        prompt = build_detection_prompt(dets, "Q?")
        assert "person" in prompt
        assert "conf 0.90" in prompt

    def test_horizontal_person_annotated(self) -> None:
        """A horizontal person bbox gets an orientation annotation."""
        dets = [_det("person", 0.9, 0, 390, 520, 470)]  # wide, short -> horizontal
        prompt = build_detection_prompt(dets, "Q?")
        assert "horizontal orientation" in prompt

    def test_vertical_person_not_annotated(self) -> None:
        """A vertical (taller than wide) person bbox has no orientation annotation."""
        dets = [_det("person", 0.9, 200, 50, 300, 450)]  # tall, narrow
        prompt = build_detection_prompt(dets, "Q?")
        assert "horizontal orientation" not in prompt

    def test_two_persons_overlapping(self) -> None:
        """Two overlapping person boxes are described as overlapping."""
        dets = [
            _det("person", 0.9, 80, 40, 360, 480),
            _det("person", 0.9, 200, 30, 500, 480),
        ]
        prompt = build_detection_prompt(dets, "Q?")
        assert "Person bounding boxes: overlapping" in prompt

    def test_two_persons_non_overlapping(self) -> None:
        """Two disjoint person boxes are described as non-overlapping."""
        dets = [
            _det("person", 0.9, 10, 50, 200, 480),
            _det("person", 0.9, 440, 50, 630, 480),
        ]
        prompt = build_detection_prompt(dets, "Q?")
        assert "Person bounding boxes: non-overlapping" in prompt

    def test_person_and_animal_overlap_described(self) -> None:
        """A person and animal box are compared for overlap."""
        dets = [
            _det("person", 0.9, 0, 0, 200, 200),
            _det("dog", 0.8, 50, 50, 150, 150),
        ]
        prompt = build_detection_prompt(dets, "Q?")
        assert "Animal bounding box:" in prompt

    def test_no_animal_no_animal_line(self) -> None:
        """No animal detections means no animal overlap line."""
        dets = [_det("person", 0.9, 0, 0, 100, 100)]
        prompt = build_detection_prompt(dets, "Q?")
        assert "Animal bounding box:" not in prompt

    def test_extra_lines_inserted_before_question(self) -> None:
        """extra_lines (e.g. audio transcript) appear before the trailing question."""
        prompt = build_detection_prompt(
            [], "Is this violent?", extra_lines=["Audio: shouting, glass breaking"]
        )
        audio_idx = prompt.index("Audio: shouting")
        question_idx = prompt.rindex("Is this violent?")
        assert audio_idx < question_idx

    def test_no_extra_lines_by_default(self) -> None:
        """Without extra_lines, no additional context lines are added."""
        prompt = build_detection_prompt([], "Q?")
        assert "Audio:" not in prompt

    def test_default_animal_labels_excludes_bird(self) -> None:
        """A 'bird' detection is not treated as an animal under the default label set."""
        dets = [
            _det("person", 0.9, 0, 0, 200, 200),
            _det("bird", 0.8, 50, 50, 150, 150),
        ]
        prompt = build_detection_prompt(dets, "Q?")
        assert "Animal bounding box:" not in prompt

    def test_custom_animal_labels_includes_bird(self) -> None:
        """A custom animal_labels set can broaden what counts as an animal."""
        dets = [
            _det("person", 0.9, 0, 0, 200, 200),
            _det("bird", 0.8, 50, 50, 150, 150),
        ]
        prompt = build_detection_prompt(dets, "Q?", animal_labels=frozenset({"bird"}))
        assert "Animal bounding box:" in prompt


@pytest.mark.unit
class TestBuildPayload:
    """Tests for build_payload()."""

    def test_no_template_prepends_system_raw(self) -> None:
        """Without a template, system + newline + prompt are concatenated."""
        payload = build_payload("hello", 64, "sys", None)
        assert payload["prompt"] == "sys\nhello"
        assert payload["n_predict"] == 64

    def test_no_template_no_system_uses_prompt_only(self) -> None:
        """Without a template and an empty system prompt, only the prompt is used."""
        payload = build_payload("hello", 64, "", None)
        assert payload["prompt"] == "hello"

    def test_with_template_substitutes_placeholders(self) -> None:
        """A template's {system}/{user} placeholders are substituted."""
        payload = build_payload("hello", 32, "sys", CHATML)
        assert "sys" in payload["prompt"]
        assert "hello" in payload["prompt"]
        assert "<|im_start|>assistant" in payload["prompt"]
