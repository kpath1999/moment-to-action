"""Integration tests for MobileCLIP zero-shot classification pipeline.

Tests the PreprocessorStage → MobileCLIPStage pipeline.
Note: MobileCLIP model is 379MB, so marked as @pytest.mark.slow
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from moment_to_action.hardware import ComputeBackend
from moment_to_action.messages import ClassificationMessage, FrameTensorMessage
from moment_to_action.models import ModelManager
from moment_to_action.sensors import FileImageSensor
from moment_to_action.stages.video import PreprocessorStage
from moment_to_action.stages.vlm import MobileCLIPStage

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
@pytest.mark.slow
def test_mobileclip_pipeline(
    test_image_path: Path,
) -> None:
    """Test MobileCLIP zero-shot classification pipeline end-to-end.

    Loads pedestrian.jpg, preprocesses for MobileCLIP, and runs
    zero-shot classification against a set of text prompts.

    Prompts used:
    - "a person grabbing a collar"
    - "a dog playing"
    - "a cat sleeping"
    - "an outdoor scene"

    Asserts:
    - ClassificationMessage returned
    - Label is a non-empty string
    - Confidence is float in [0, 1]
    - All scores in all_scores are in [0, 1]
    - Latency > 0 ms
    """
    # Load image
    sensor = FileImageSensor(test_image_path)
    sensor.open()
    raw_msg = sensor.read()
    sensor.close()

    # Preprocess for MobileCLIP (256x256, no normalization as per config)
    preprocess_stage = PreprocessorStage(
        target_size=(256, 256),
        letterbox=False,
        channels_first=True,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    tensor_msg = preprocess_stage.process(raw_msg, stage_idx=0)
    assert tensor_msg is not None, "Preprocessing should succeed"

    # Run MobileCLIP — stage resolves its own model path via ModelManager.
    text_prompts = [
        "a person grabbing a collar",
        "a dog playing",
        "a cat sleeping",
        "an outdoor scene",
    ]
    backend = ComputeBackend()
    manager = ModelManager()
    mobileclip_stage = MobileCLIPStage(
        text_prompts=text_prompts,
        backend=backend,
        manager=manager,
    )
    classification_msg = mobileclip_stage.process(tensor_msg, stage_idx=1)

    # Assertions
    assert classification_msg is not None, "MobileCLIPStage should return a result"
    assert isinstance(classification_msg, ClassificationMessage), (
        f"Result should be ClassificationMessage, got {type(classification_msg).__name__}"
    )

    # Check label
    assert isinstance(classification_msg.label, str), "Label should be a string"
    assert len(classification_msg.label) > 0, "Label should be non-empty"
    assert classification_msg.label in text_prompts, (
        f"Label '{classification_msg.label}' should be one of {text_prompts}"
    )

    # Check confidence
    assert isinstance(classification_msg.confidence, float), "Confidence should be float"
    assert 0.0 <= classification_msg.confidence <= 1.0, (
        f"Confidence should be in [0, 1], got {classification_msg.confidence}"
    )

    # Check all_scores
    assert isinstance(classification_msg.all_scores, dict), "all_scores should be a dict"
    assert len(classification_msg.all_scores) == len(text_prompts), (
        f"all_scores should have {len(text_prompts)} entries"
    )
    for prompt, score in classification_msg.all_scores.items():
        assert prompt in text_prompts, f"Score key '{prompt}' not in prompts"
        assert isinstance(score, float), f"Score for '{prompt}' should be float"
        assert 0.0 <= score <= 1.0, f"Score for '{prompt}' should be in [0, 1], got {score}"

    # Check latency
    assert classification_msg.latency_ms > 0, "Latency should be > 0"


@pytest.mark.integration
@pytest.mark.slow
def test_mobileclip_swappable_prompts(
    test_image_path: Path,
) -> None:
    """Test that MobileCLIP prompts can be updated at runtime.

    Creates a stage, classifies with initial prompts, then swaps prompts
    and reclassifies on the same image.

    Asserts:
    - Both classifications succeed
    - Second classification uses new prompts
    - Labels differ between the two runs (or at minimum are computed correctly)
    """
    # Load and preprocess image
    sensor = FileImageSensor(test_image_path)
    sensor.open()
    raw_msg = sensor.read()
    sensor.close()

    preprocess_stage = PreprocessorStage(
        target_size=(256, 256),
        letterbox=False,
        channels_first=True,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    tensor_msg = preprocess_stage.process(raw_msg, stage_idx=0)
    assert tensor_msg is not None

    # First classification — stage resolves its own model path via ModelManager.
    initial_prompts = ["a person", "an animal", "a landscape"]
    backend = ComputeBackend()
    manager = ModelManager()
    mobileclip_stage = MobileCLIPStage(
        text_prompts=initial_prompts,
        backend=backend,
        manager=manager,
    )
    result1 = mobileclip_stage.process(tensor_msg, stage_idx=1)
    assert result1 is not None
    assert isinstance(result1, ClassificationMessage)
    assert result1.label in initial_prompts

    # Swap prompts
    new_prompts = ["a violent scene", "a peaceful scene", "a chaotic scene"]
    mobileclip_stage.update_prompts(new_prompts)

    # Second classification with same image, different prompts
    result2 = mobileclip_stage.process(tensor_msg, stage_idx=1)
    assert result2 is not None
    assert isinstance(result2, ClassificationMessage)
    assert result2.label in new_prompts, (
        f"After update_prompts, label should be in {new_prompts}, got {result2.label}"
    )

    # Verify scores are for new prompts
    assert set(result2.all_scores.keys()) == set(new_prompts), (
        "all_scores should contain new prompts"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_mobileclip_preprocessing_shapes(
    test_image_path: Path,
) -> None:
    """Test that preprocessing produces correct tensor shapes for MobileCLIP.

    MobileCLIP expects [1, 3, 256, 256] float32 tensors, channels-first.

    Asserts:
    - Tensor shape is [1, 3, 256, 256]
    - Tensor dtype is float32
    - Tensor values are reasonable (uint8 in 0-255 range before normalization)
    """
    # Load image
    sensor = FileImageSensor(test_image_path)
    sensor.open()
    raw_msg = sensor.read()
    sensor.close()

    # Preprocess for MobileCLIP
    preprocess_stage = PreprocessorStage(
        target_size=(256, 256),
        letterbox=False,
        channels_first=True,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    tensor_msg = preprocess_stage.process(raw_msg, stage_idx=0)

    assert tensor_msg is not None
    assert isinstance(tensor_msg, FrameTensorMessage)

    # Check shape
    assert tensor_msg.tensor.shape == (1, 3, 256, 256), (
        f"Tensor shape should be (1, 3, 256, 256), got {tensor_msg.tensor.shape}"
    )

    # Check dtype
    import numpy as np

    assert tensor_msg.tensor.dtype == np.float32, (
        f"Tensor dtype should be float32, got {tensor_msg.tensor.dtype}"
    )

    # Check value range (with mean=0, std=1, values should be in 0-255 range for uint8 images)
    assert tensor_msg.tensor.min() >= 0.0, "Min value should be >= 0"
    assert tensor_msg.tensor.max() <= 255.0, "Max value should be <= 255"
