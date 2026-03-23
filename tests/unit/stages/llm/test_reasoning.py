"""Unit tests for ReasoningStage.

Tests LLM reasoning in stub mode with DetectionMessage → ReasoningMessage.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from moment_to_action.messages.llm import ReasoningMessage
from moment_to_action.messages.video import BoundingBox, DetectionMessage
from moment_to_action.stages.llm._reasoning import ReasoningStage


@pytest.mark.unit
class TestReasoningStage:
    """Tests for ReasoningStage."""

    @pytest.fixture
    def sample_detection_message(self) -> DetectionMessage:
        """Create a sample detection message with multiple boxes."""
        boxes = [
            BoundingBox(
                x1=100.0,
                y1=150.0,
                x2=500.0,
                y2=600.0,
                confidence=0.95,
                class_id=0,
                label="person",
            ),
            BoundingBox(
                x1=50.0,
                y1=200.0,
                x2=150.0,
                y2=400.0,
                confidence=0.87,
                class_id=1,
                label="hand",
            ),
            BoundingBox(
                x1=300.0,
                y1=250.0,
                x2=400.0,
                y2=350.0,
                confidence=0.72,
                class_id=2,
                label="face",
            ),
            BoundingBox(
                x1=200.0,
                y1=100.0,
                x2=350.0,
                y2=500.0,
                confidence=0.65,
                class_id=0,
                label="person",
            ),
            BoundingBox(
                x1=450.0,
                y1=300.0,
                x2=550.0,
                y2=450.0,
                confidence=0.58,
                class_id=1,
                label="hand",
            ),
            BoundingBox(
                x1=600.0,
                y1=500.0,
                x2=700.0,
                y2=600.0,
                confidence=0.42,
                class_id=3,
                label="phone",
            ),
        ]
        return DetectionMessage(
            boxes=boxes,
            timestamp=time.time(),
        )

    def test_reasoning_stage_stub_mode_initialization(self) -> None:
        """Test ReasoningStage initialization in stub mode (no model)."""
        stage = ReasoningStage(model_path=None)

        assert stage._handle is None
        assert stage._system_prompt is not None
        assert len(stage._system_prompt) > 0

    def test_reasoning_stage_stub_mode_full(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test ReasoningStage in stub mode: initialization and processing.

        Covers initialization without model_path, verifying backend is None,
        handle is None, and stub response is generated correctly.
        """
        # Test 1: Initialize without error
        stage = ReasoningStage()

        # Test 2: Verify _backend is None
        assert stage._backend is None

        # Test 3: Verify _handle is None
        assert stage._handle is None

        # Test 4: Running in stub mode returns ReasoningMessage with stub response
        result = stage.process(sample_detection_message)

        assert isinstance(result, ReasoningMessage)
        assert "[LLM stub]" in result.response
        assert "chars" in result.response.lower()

    def test_reasoning_stage_default_system_prompt(self) -> None:
        """Test ReasoningStage uses default system prompt when not provided."""
        stage = ReasoningStage(model_path=None)

        assert "analyzing detections" in stage._system_prompt.lower()
        assert "wearable device" in stage._system_prompt.lower()

    def test_reasoning_stage_custom_system_prompt(self) -> None:
        """Test ReasoningStage uses custom system prompt when provided."""
        custom_prompt = "You are a robot analyzing scenes."
        stage = ReasoningStage(model_path=None, system_prompt=custom_prompt)

        assert stage._system_prompt == custom_prompt

    def test_build_prompt_includes_system_prompt(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that _build_prompt includes system prompt at the beginning."""
        custom_prompt = "Custom system message"
        stage = ReasoningStage(model_path=None, system_prompt=custom_prompt)

        prompt = stage._build_prompt(sample_detection_message)

        assert prompt.startswith(custom_prompt)

    def test_build_prompt_includes_detections(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that _build_prompt includes detection information."""
        stage = ReasoningStage(model_path=None)

        prompt = stage._build_prompt(sample_detection_message)

        assert "Detections:" in prompt
        assert "person" in prompt
        assert "hand" in prompt
        assert "face" in prompt

    def test_build_prompt_includes_confidence_scores(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that _build_prompt includes confidence scores in detection output."""
        stage = ReasoningStage(model_path=None)

        prompt = stage._build_prompt(sample_detection_message)

        # Check for confidence values (formatted to 2 decimals)
        assert "0.95" in prompt or "0.95" in prompt
        assert "confidence:" in prompt.lower()

    def test_build_prompt_includes_bounding_box_coordinates(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that _build_prompt includes bounding box coordinates."""
        stage = ReasoningStage(model_path=None)

        prompt = stage._build_prompt(sample_detection_message)

        # Check for coordinate information in format [x1, y1, x2, y2]
        assert "position:" in prompt.lower()
        assert "[" in prompt
        assert "]" in prompt

    def test_build_prompt_uses_top_5_detections(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that _build_prompt uses top 5 detections by confidence."""
        stage = ReasoningStage(model_path=None)

        prompt = stage._build_prompt(sample_detection_message)

        # Count detection entries (lines starting with "  - ")
        detection_lines = [line for line in prompt.split("\n") if line.strip().startswith("-")]

        # Should have exactly 5 detections (or fewer if less than 5 available)
        assert len(detection_lines) == min(5, len(sample_detection_message.boxes))

    def test_build_prompt_detections_ordered_by_confidence(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that top 5 detections are sorted by confidence (descending)."""
        stage = ReasoningStage(model_path=None)

        prompt = stage._build_prompt(sample_detection_message)

        # Extract confidence scores from prompt
        lines = prompt.split("\n")
        confidences = []
        for line in lines:
            if "confidence:" in line.lower():
                # Extract confidence value (e.g., "0.95")
                try:
                    score_str = line.split("confidence:")[-1].split(",")[0].strip()
                    confidences.append(float(score_str))
                except (ValueError, IndexError):
                    pass

        # Verify confidences are in descending order
        if len(confidences) > 1:
            for i in range(len(confidences) - 1):
                assert confidences[i] >= confidences[i + 1]

    def test_build_prompt_includes_question(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that _build_prompt includes the scene analysis question."""
        stage = ReasoningStage(model_path=None)

        prompt = stage._build_prompt(sample_detection_message)

        assert "What is happening in this scene?" in prompt

    def test_reasoning_stage_stub_response_format(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that stub response includes '[LLM stub]' marker and char count."""
        stage = ReasoningStage(model_path=None)

        result = stage.process(sample_detection_message)

        assert isinstance(result, ReasoningMessage)
        assert "[LLM stub]" in result.response
        assert "chars" in result.response.lower()

    def test_reasoning_stage_stub_response_includes_char_count(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that stub response includes the character count of the prompt."""
        stage = ReasoningStage(model_path=None)

        result = stage.process(sample_detection_message)

        # Extract char count from response
        assert result is not None
        assert isinstance(result, ReasoningMessage)
        response_text = result.response
        assert "[LLM stub]" in response_text

        # The response should mention the prompt length
        assert str(len(result.prompt)) in response_text

    def test_process_returns_reasoning_message(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that _process() returns ReasoningMessage with correct structure."""
        stage = ReasoningStage(model_path=None)

        result = stage.process(sample_detection_message)

        assert isinstance(result, ReasoningMessage)
        assert hasattr(result, "response")
        assert hasattr(result, "prompt")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "latency_ms")

    def test_reasoning_message_contains_prompt(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that ReasoningMessage contains the exact prompt sent to LLM."""
        stage = ReasoningStage(model_path=None)

        result = stage.process(sample_detection_message)

        assert result is not None
        assert isinstance(result, ReasoningMessage)
        assert result.prompt is not None
        assert len(result.prompt) > 0
        # Prompt should contain system prompt and detections
        assert stage._system_prompt in result.prompt

    def test_reasoning_message_preserves_timestamp(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that ReasoningMessage preserves timestamp from input."""
        stage = ReasoningStage(model_path=None)

        result = stage.process(sample_detection_message)

        assert result is not None
        assert isinstance(result, ReasoningMessage)
        assert result.timestamp == sample_detection_message.timestamp

    def test_reasoning_stage_rejects_non_detection_message(self) -> None:
        """Test that ReasoningStage rejects non-DetectionMessage input."""
        from moment_to_action.messages.sensor import RawFrameMessage

        stage = ReasoningStage(model_path=None)

        wrong_msg = RawFrameMessage(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=time.time(),
            width=640,
            height=480,
        )

        with pytest.raises(TypeError, match="expects DetectionMessage"):
            stage.process(wrong_msg)

    def test_reasoning_stage_with_empty_detections(self) -> None:
        """Test ReasoningStage with DetectionMessage containing no boxes."""
        stage = ReasoningStage(model_path=None)

        msg = DetectionMessage(
            boxes=[],
            timestamp=time.time(),
        )

        result = stage.process(msg)

        assert isinstance(result, ReasoningMessage)
        assert "Detections:" in result.prompt
        # Should still generate response even with no detections
        assert len(result.response) > 0

    def test_reasoning_stage_with_single_detection(self) -> None:
        """Test ReasoningStage with DetectionMessage containing single box."""
        stage = ReasoningStage(model_path=None)

        boxes = [
            BoundingBox(
                x1=100.0,
                y1=150.0,
                x2=500.0,
                y2=600.0,
                confidence=0.95,
                class_id=0,
                label="person",
            )
        ]
        msg = DetectionMessage(
            boxes=boxes,
            timestamp=time.time(),
        )

        result = stage.process(msg)

        assert isinstance(result, ReasoningMessage)
        assert "person" in result.prompt
        assert "0.95" in result.prompt

    def test_reasoning_stage_name(self) -> None:
        """Test that stage name is correct."""
        stage = ReasoningStage(model_path=None)

        assert stage.name == "ReasoningStage"

    def test_reasoning_stage_latency_stamped(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that latency_ms is stamped on the result."""
        stage = ReasoningStage(model_path=None)

        result = stage.process(sample_detection_message)

        assert result is not None
        assert isinstance(result, ReasoningMessage)
        assert result.latency_ms >= 0.0

    def test_system_prompt_consistency_across_calls(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that system prompt is consistent across multiple calls."""
        custom_prompt = "Analyze the scene carefully."
        stage = ReasoningStage(model_path=None, system_prompt=custom_prompt)

        result1 = stage.process(sample_detection_message)
        result2 = stage.process(sample_detection_message)

        assert result1 is not None
        assert isinstance(result1, ReasoningMessage)
        assert result2 is not None
        assert isinstance(result2, ReasoningMessage)
        assert custom_prompt in result1.prompt
        assert custom_prompt in result2.prompt

    def test_build_prompt_with_low_confidence_detections(self) -> None:
        """Test _build_prompt with detections having low confidence scores."""
        stage = ReasoningStage(model_path=None)

        boxes = [
            BoundingBox(
                x1=100.0,
                y1=150.0,
                x2=500.0,
                y2=600.0,
                confidence=0.1,
                class_id=0,
                label="person",
            )
        ]
        msg = DetectionMessage(
            boxes=boxes,
            timestamp=time.time(),
        )

        prompt = stage._build_prompt(msg)

        assert "person" in prompt
        assert "0.10" in prompt

    def test_prompt_formatting_structure(self, sample_detection_message: DetectionMessage) -> None:
        """Test that prompt has proper formatting with lines and structure."""
        stage = ReasoningStage(model_path=None)

        prompt = stage._build_prompt(sample_detection_message)

        lines = prompt.split("\n")

        # Should have multiple lines
        assert len(lines) > 1

        # Should have proper structure: system prompt, blank line, detections section, etc.
        assert any("Detections:" in line for line in lines)
        assert any("What is happening" in line for line in lines)

    def test_detection_format_in_prompt(self, sample_detection_message: DetectionMessage) -> None:
        """Test that each detection is formatted correctly in the prompt."""
        stage = ReasoningStage(model_path=None)

        prompt = stage._build_prompt(sample_detection_message)

        assert "  - " in prompt
        assert "confidence:" in prompt.lower()
        assert "position:" in prompt.lower()

    def test_reasoning_stage_with_model_path_mocked(self) -> None:
        """Test ReasoningStage initialisation with a model path (mocked backend).

        Covers lines 42-44 — the if-model_path branch that constructs a
        ComputeBackend and loads the model.  The ComputeBackend is mocked so
        no real model file is needed.
        """
        from unittest.mock import MagicMock, patch

        mock_backend = MagicMock()
        mock_handle = MagicMock()
        mock_backend.load_model.return_value = mock_handle

        with patch(
            "moment_to_action.stages.llm._reasoning.ComputeBackend",
            return_value=mock_backend,
        ):
            stage = ReasoningStage(model_path="/fake/model.onnx")

        # Backend and handle should be set (not stub mode).
        assert stage._backend is mock_backend
        assert stage._handle is mock_handle
        mock_backend.load_model.assert_called_once_with("/fake/model.onnx")
