"""Shared synthetic benchmark scenes for the LLM and VLM benches.

Fixture data — not library API. ``Scene.audio_transcript`` is only consumed by
the LLM benchmark (as an extra prompt context line); the VLM benchmark ignores
it since it renders frames directly and has no text-audio input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from moment_to_action.models.image.detection._types import BoundingBox, Detection


@dataclass(frozen=True)
class Scene:
    """One benchmark scene backed by YOLO-realistic inputs.

    Attributes:
        name: Short identifier used in output.
        app: Target application name.
        task: The binary/multi-label question the system asks.
        detections: YOLO detections (label + confidence + bbox) used both to
            derive the LLM's prompt context and to render the VLM's synthetic frame.
        audio_transcript: Transcript from an audio model, used only by the LLM
            benchmark as extra prompt context. ``None`` for apps that do not use
            audio.
        expected_label: Correct answer token (e.g. "YES", "NO", "COMPLIANT").
        recall_keywords: Words expected from a correct answer. Labels that
            appear verbatim in ``detections`` are excluded so that input-echoing
            does not inflate recall.
    """

    name: str
    app: str
    task: str
    detections: list[Detection]
    audio_transcript: str | None
    expected_label: str
    recall_keywords: list[str] = field(default_factory=list)


def _bb(x1: int, y1: int, x2: int, y2: int) -> BoundingBox:
    """Shorthand BoundingBox constructor.

    Args:
        x1: Left edge.
        y1: Top edge.
        x2: Right edge.
        y2: Bottom edge.

    Returns:
        BoundingBox instance.
    """
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _det(label: str, conf: float, x1: int, y1: int, x2: int, y2: int) -> Detection:
    """Shorthand Detection constructor.

    Args:
        label: Class label.
        conf: Confidence score.
        x1: Left edge.
        y1: Top edge.
        x2: Right edge.
        y2: Bottom edge.

    Returns:
        Detection instance.
    """
    return Detection(label=label, confidence=conf, bbox=_bb(x1, y1, x2, y2))


SCENES: list[Scene] = [
    # --- Violence Detection -------------------------------------------------
    # Positive: two persons with heavily overlapping bboxes, audio confirms altercation
    Scene(
        name="violence_fight",
        app="violence_detection",
        task="Is a violent incident occurring? Answer YES or NO, then one sentence of reasoning.",
        detections=[
            _det("person", 0.95, 80, 40, 360, 480),
            _det("person", 0.92, 200, 30, 500, 480),  # large overlap with first person
        ],
        audio_transcript="shouting, impact sounds, glass breaking",
        expected_label="YES",
        recall_keywords=["yes", "fight", "violen", "aggress", "altercation", "physical"],
    ),
    # Negative: two persons at opposite sides of frame, no overlap, calm audio
    Scene(
        name="violence_calm",
        app="violence_detection",
        task="Is a violent incident occurring? Answer YES or NO, then one sentence of reasoning.",
        detections=[
            _det("person", 0.93, 10, 50, 200, 480),  # left side
            _det("person", 0.90, 440, 50, 630, 480),  # right side, no overlap
        ],
        audio_transcript="ambient music, quiet conversation, laughter",
        expected_label="NO",
        recall_keywords=["no", "calm", "peaceful", "safe", "non-violent", "normal"],
    ),
    # --- Fall Detection -----------------------------------------------------
    # Positive: person bbox is horizontal (width >> height), located at bottom of frame
    Scene(
        name="fall_detected",
        app="fall_detection",
        task="Has a person fallen? Answer YES or NO, then one sentence of reasoning.",
        detections=[
            _det("person", 0.91, 50, 390, 520, 470),  # horizontal (w=470 > h=80), bottom frame
            _det("chair", 0.74, 300, 200, 500, 400),
        ],
        audio_transcript=None,
        expected_label="YES",
        recall_keywords=["yes", "fall", "fallen", "ground", "floor", "horizontal", "lying"],
    ),
    # Negative: person bbox is vertical (height >> width), centered in frame
    Scene(
        name="fall_standing",
        app="fall_detection",
        task="Has a person fallen? Answer YES or NO, then one sentence of reasoning.",
        detections=[
            _det("person", 0.95, 220, 40, 400, 480),  # vertical (w=180 < h=440), mid-center
            _det("desk", 0.81, 400, 200, 640, 480),
            _det("monitor", 0.78, 460, 60, 620, 260),
        ],
        audio_transcript=None,
        expected_label="NO",
        recall_keywords=["no", "standing", "upright", "vertical", "normal", "not fallen"],
    ),
    # --- Animal Threat / Attack Detection ----------------------------------
    # Positive: dog bbox overlaps heavily with person bbox, audio confirms aggression
    Scene(
        name="animal_threat",
        app="animal_threat_detection",
        task=(
            "Is an animal posing an immediate threat to a person? "
            "Answer YES or NO, then one sentence of reasoning."
        ),
        detections=[
            _det("person", 0.93, 150, 80, 430, 480),
            _det("dog", 0.88, 350, 180, 620, 480),  # overlaps with person bbox
        ],
        audio_transcript="aggressive barking, growling",
        expected_label="YES",
        recall_keywords=["yes", "threat", "danger", "aggress", "attack", "immediate"],
    ),
    # Negative: dog bbox small and far from person (no overlap), calm audio
    Scene(
        name="animal_safe",
        app="animal_threat_detection",
        task=(
            "Is an animal posing an immediate threat to a person? "
            "Answer YES or NO, then one sentence of reasoning."
        ),
        detections=[
            _det("person", 0.94, 80, 50, 380, 480),  # foreground, left
            _det("dog", 0.76, 530, 320, 610, 400),  # small (background), right, no overlap
        ],
        audio_transcript="ambient park sounds, distant barking",
        expected_label="NO",
        recall_keywords=["no", "safe", "distant", "no threat", "away", "not immediate"],
    ),
    # --- Eating Detection (egocentric wearable) ----------------------------
    # Positive: food items dominate foreground (large bbox area = close to camera)
    Scene(
        name="eating_yes",
        app="eating_detection",
        task=(
            "Egocentric view from wearable camera. "
            "Is the wearer currently eating or drinking? "
            "Answer YES or NO, then one sentence of reasoning."
        ),
        detections=[
            _det("fork", 0.89, 240, 300, 400, 440),  # foreground
            _det("sandwich", 0.84, 140, 270, 450, 460),  # foreground
            _det("plate", 0.91, 70, 260, 580, 470),  # large, foreground
            _det("dining table", 0.72, 0, 410, 640, 480),  # background strip
        ],
        audio_transcript=None,
        expected_label="YES",
        recall_keywords=["yes", "eating", "meal", "consuming", "food", "fork"],
    ),
    # Negative: computer peripherals dominate foreground, food present but background
    Scene(
        name="eating_no",
        app="eating_detection",
        task=(
            "Egocentric view from wearable camera. "
            "Is the wearer currently eating or drinking? "
            "Answer YES or NO, then one sentence of reasoning."
        ),
        detections=[
            _det("keyboard", 0.93, 90, 360, 550, 470),  # foreground
            _det("laptop", 0.88, 140, 200, 500, 400),  # midground
            _det("monitor", 0.85, 40, 40, 600, 300),  # large background
            _det("cup", 0.65, 575, 360, 635, 440),  # small, right corner
        ],
        audio_transcript=None,
        expected_label="NO",
        recall_keywords=["no", "working", "typing", "not eating", "computer", "keyboard"],
    ),
    # --- Workplace Safety / PPE Compliance ---------------------------------
    # Positive: all required PPE items detected on or near the person
    Scene(
        name="ppe_compliant",
        app="ppe_compliance",
        task=(
            "Is the construction worker wearing all required PPE "
            "(hard hat, safety vest, gloves, boots)? "
            "Answer COMPLIANT or NON-COMPLIANT, then list present and missing items."
        ),
        detections=[
            _det("person", 0.96, 120, 40, 520, 480),
            _det("hard hat", 0.91, 230, 40, 420, 140),  # top of frame, on head
            _det("safety vest", 0.88, 140, 150, 500, 340),
            _det("glove", 0.79, 120, 310, 230, 420),
            _det("glove", 0.77, 410, 310, 520, 420),
            _det("boot", 0.83, 160, 410, 290, 480),
            _det("boot", 0.80, 350, 410, 480, 480),
        ],
        audio_transcript=None,
        expected_label="COMPLIANT",
        recall_keywords=["compliant", "hat", "vest", "glove", "boot", "all", "present"],
    ),
    # Negative: hard hat and gloves absent from detections
    Scene(
        name="ppe_violation",
        app="ppe_compliance",
        task=(
            "Is the construction worker wearing all required PPE "
            "(hard hat, safety vest, gloves, boots)? "
            "Answer COMPLIANT or NON-COMPLIANT, then list present and missing items."
        ),
        detections=[
            _det("person", 0.95, 120, 40, 520, 480),
            _det("safety vest", 0.90, 140, 150, 500, 340),
            _det("boot", 0.84, 160, 410, 290, 480),
            _det("boot", 0.82, 350, 410, 480, 480),
            # hard hat and gloves absent
        ],
        audio_transcript=None,
        expected_label="NON-COMPLIANT",
        recall_keywords=["non-compliant", "missing", "hat", "glove", "absent", "violation"],
    ),
]
