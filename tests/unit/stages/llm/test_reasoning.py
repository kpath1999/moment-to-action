"""Unit tests for ReasoningStage.

Tests LLM reasoning in stub mode with DetectionMessage → ReasoningMessage.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.messages import DetectionMessage
from moment_to_action.messages.llm import ReasoningMessage
from moment_to_action.models.image.detection._types import BoundingBox, Detection
from moment_to_action.stages.llm._reasoning import ReasoningStage


def _det(
    label: str,
    confidence: float,
    x1: float = 0.0,
    y1: float = 0.0,
    x2: float = 100.0,
    y2: float = 100.0,
) -> Detection:
    """Build a Detection with the given fields."""
    return Detection(
        label=label, confidence=confidence, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    )


@pytest.mark.unit
class TestReasoningStage:
    """Tests for ReasoningStage."""

    @pytest.fixture
    def sample_detection_message(self) -> DetectionMessage:
        """Create a sample detection message with multiple detections."""
        detections = [
            _det("person", 0.95, x1=100.0, y1=150.0, x2=500.0, y2=600.0),
            _det("hand", 0.87, x1=50.0, y1=200.0, x2=150.0, y2=400.0),
            _det("face", 0.72, x1=300.0, y1=250.0, x2=400.0, y2=350.0),
            _det("person", 0.65, x1=200.0, y1=100.0, x2=350.0, y2=500.0),
            _det("hand", 0.58, x1=450.0, y1=300.0, x2=550.0, y2=450.0),
            _det("phone", 0.42, x1=600.0, y1=500.0, x2=700.0, y2=600.0),
        ]
        return DetectionMessage(detections=detections, timestamp=time.time())

    def test_reasoning_stage_stub_mode_initialization(self) -> None:
        """Test ReasoningStage initialization in stub mode (no model)."""
        stage = ReasoningStage()

        assert stage._handle is None
        assert stage._system_prompt is not None
        assert len(stage._system_prompt) > 0

    def test_reasoning_stage_stub_mode_full(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test ReasoningStage in stub mode: initialization and processing."""
        stage = ReasoningStage()

        assert stage._platform is None
        assert stage._handle is None

        result = stage.process(sample_detection_message)

        assert isinstance(result, ReasoningMessage)
        assert "[LLM stub]" in result.response
        assert "chars" in result.response.lower()

    def test_reasoning_stage_default_system_prompt(self) -> None:
        """Test ReasoningStage uses default system prompt when not provided."""
        stage = ReasoningStage()

        assert "analyzing detections" in stage._system_prompt.lower()
        assert "wearable device" in stage._system_prompt.lower()

    def test_reasoning_stage_custom_system_prompt(self) -> None:
        """Test ReasoningStage uses custom system prompt when provided."""
        custom_prompt = "You are a robot analyzing scenes."
        stage = ReasoningStage(system_prompt=custom_prompt)

        assert stage._system_prompt == custom_prompt

    def test_build_prompt_includes_system_prompt(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that _build_prompt includes system prompt at the beginning."""
        custom_prompt = "Custom system message"
        stage = ReasoningStage(system_prompt=custom_prompt)

        prompt = stage._build_prompt(sample_detection_message)

        assert prompt.startswith(custom_prompt)

    def test_build_prompt_includes_detections(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that _build_prompt includes detection information."""
        stage = ReasoningStage()

        prompt = stage._build_prompt(sample_detection_message)

        assert "Detections:" in prompt
        assert "person" in prompt
        assert "hand" in prompt
        assert "face" in prompt

    def test_build_prompt_includes_confidence_scores(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that _build_prompt includes confidence scores in detection output."""
        stage = ReasoningStage()

        prompt = stage._build_prompt(sample_detection_message)

        assert "0.95" in prompt
        assert "confidence:" in prompt.lower()

    def test_build_prompt_includes_bounding_box_coordinates(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that _build_prompt includes bounding box coordinates."""
        stage = ReasoningStage()

        prompt = stage._build_prompt(sample_detection_message)

        assert "position:" in prompt.lower()
        assert "[" in prompt
        assert "]" in prompt

    def test_build_prompt_uses_top_5_detections(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that _build_prompt uses top 5 detections by confidence."""
        stage = ReasoningStage()

        prompt = stage._build_prompt(sample_detection_message)

        detection_lines = [line for line in prompt.split("\n") if line.strip().startswith("-")]

        assert len(detection_lines) == min(5, len(sample_detection_message.detections))

    def test_build_prompt_detections_ordered_by_confidence(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that top 5 detections are sorted by confidence (descending)."""
        stage = ReasoningStage()

        prompt = stage._build_prompt(sample_detection_message)

        lines = prompt.split("\n")
        confidences = []
        for line in lines:
            if "confidence:" in line.lower():
                try:
                    score_str = line.split("confidence:")[-1].split(",")[0].strip()
                    confidences.append(float(score_str))
                except (ValueError, IndexError):
                    pass

        if len(confidences) > 1:
            for i in range(len(confidences) - 1):
                assert confidences[i] >= confidences[i + 1]

    def test_build_prompt_includes_question(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that _build_prompt includes the scene analysis question."""
        stage = ReasoningStage()

        prompt = stage._build_prompt(sample_detection_message)

        assert "What is happening in this scene?" in prompt

    def test_reasoning_stage_stub_response_format(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that stub response includes '[LLM stub]' marker and char count."""
        stage = ReasoningStage()

        result = stage.process(sample_detection_message)

        assert isinstance(result, ReasoningMessage)
        assert "[LLM stub]" in result.response
        assert "chars" in result.response.lower()

    def test_reasoning_stage_stub_response_includes_char_count(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that stub response includes the character count of the prompt."""
        stage = ReasoningStage()

        result = stage.process(sample_detection_message)

        assert result is not None
        assert isinstance(result, ReasoningMessage)
        assert "[LLM stub]" in result.response
        assert str(len(result.prompt)) in result.response

    def test_process_returns_reasoning_message(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that _process() returns ReasoningMessage with correct structure."""
        stage = ReasoningStage()

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
        stage = ReasoningStage()

        result = stage.process(sample_detection_message)

        assert result is not None
        assert isinstance(result, ReasoningMessage)
        assert result.prompt is not None
        assert len(result.prompt) > 0
        assert stage._system_prompt in result.prompt

    def test_reasoning_message_preserves_timestamp(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that ReasoningMessage preserves timestamp from input."""
        stage = ReasoningStage()

        result = stage.process(sample_detection_message)

        assert result is not None
        assert isinstance(result, ReasoningMessage)
        assert result.timestamp == sample_detection_message.timestamp

    def test_reasoning_stage_rejects_non_detection_message(self) -> None:
        """Test that ReasoningStage rejects non-DetectionMessage input."""
        from moment_to_action.messages.sensor import RawFrameMessage

        stage = ReasoningStage()

        wrong_msg = RawFrameMessage(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=time.time(),
            width=640,
            height=480,
        )

        with pytest.raises(TypeError, match="expects DetectionMessage"):
            stage.process(wrong_msg)

    def test_reasoning_stage_with_empty_detections(self) -> None:
        """Test ReasoningStage with DetectionMessage containing no detections."""
        stage = ReasoningStage()

        msg = DetectionMessage(detections=[], timestamp=time.time())

        result = stage.process(msg)

        assert isinstance(result, ReasoningMessage)
        assert "Detections:" in result.prompt
        assert len(result.response) > 0

    def test_reasoning_stage_with_single_detection(self) -> None:
        """Test ReasoningStage with DetectionMessage containing single detection."""
        stage = ReasoningStage()

        msg = DetectionMessage(
            detections=[_det("person", 0.95, x1=100.0, y1=150.0, x2=500.0, y2=600.0)],
            timestamp=time.time(),
        )

        result = stage.process(msg)

        assert isinstance(result, ReasoningMessage)
        assert "person" in result.prompt
        assert "0.95" in result.prompt

    def test_reasoning_stage_name(self) -> None:
        """Test that stage name is correct."""
        stage = ReasoningStage()

        assert stage.name == "ReasoningStage"

    def test_reasoning_stage_latency_stamped(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that latency_ms is stamped on the result."""
        from moment_to_action.metrics import MetricsCollector

        stage = ReasoningStage()

        metrics = MetricsCollector(session_id="test_reasoning_latency")
        with metrics.start_trace():
            result = stage.process(sample_detection_message, metrics=metrics)

        assert result is not None
        assert isinstance(result, ReasoningMessage)
        assert result.latency_ms >= 0.0

    def test_system_prompt_consistency_across_calls(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """Test that system prompt is consistent across multiple calls."""
        custom_prompt = "Analyze the scene carefully."
        stage = ReasoningStage(system_prompt=custom_prompt)

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
        stage = ReasoningStage()

        msg = DetectionMessage(
            detections=[_det("person", 0.1, x1=100.0, y1=150.0, x2=500.0, y2=600.0)],
            timestamp=time.time(),
        )

        prompt = stage._build_prompt(msg)

        assert "person" in prompt
        assert "0.10" in prompt

    def test_prompt_formatting_structure(self, sample_detection_message: DetectionMessage) -> None:
        """Test that prompt has proper formatting with lines and structure."""
        stage = ReasoningStage()

        prompt = stage._build_prompt(sample_detection_message)

        lines = prompt.split("\n")

        assert len(lines) > 1
        assert any("Detections:" in line for line in lines)
        assert any("What is happening" in line for line in lines)

    def test_detection_format_in_prompt(self, sample_detection_message: DetectionMessage) -> None:
        """Test that each detection is formatted correctly in the prompt."""
        stage = ReasoningStage()

        prompt = stage._build_prompt(sample_detection_message)

        assert "  - " in prompt
        assert "confidence:" in prompt.lower()
        assert "position:" in prompt.lower()

    def test_manager_required_with_model_id(self) -> None:
        """Test that an error is thrown if a model ID is provided but not the manager."""
        from moment_to_action.models import ModelID

        with pytest.raises(ValueError, match="Model manager is required"):
            ReasoningStage(model_id=ModelID.YOLO_V8)

        with pytest.raises(ValueError, match="Model manager is required"):
            ReasoningStage(model_id=ModelID.YOLO_V8, manager=None)

    def test_reasoning_stage_with_model_id_mocked(self) -> None:
        """Test ReasoningStage initialisation with a model_id (mocked backend + manager)."""
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from moment_to_action.models import ModelID

        fake_path = Path("/fake/model.onnx")
        mock_manager = MagicMock()
        mock_manager.get_path.return_value = fake_path

        mock_platform = MagicMock()
        mock_handle = MagicMock()
        mock_platform.load_onnx.return_value = mock_handle

        with patch(
            "moment_to_action.stages.llm._reasoning.Platform",
            return_value=mock_platform,
        ):
            stage = ReasoningStage(model_id=ModelID.YOLO_V8, manager=mock_manager)

        assert stage._platform is mock_platform
        assert stage._handle is mock_handle
        mock_manager.get_path.assert_called_once_with(ModelID.YOLO_V8)
        mock_platform.load_onnx.assert_called_once_with(ComputeUnit.CPU, fake_path)

    def test_reasoning_stage_with_tflite_model_id(self) -> None:
        """ReasoningStage with a .tflite path calls platform.load_tflite."""
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from moment_to_action.models import ModelID

        fake_path = Path("/fake/model.tflite")
        mock_manager = MagicMock()
        mock_manager.get_path.return_value = fake_path

        mock_platform = MagicMock()
        mock_handle = MagicMock()
        mock_platform.load_tflite.return_value = mock_handle

        with patch(
            "moment_to_action.stages.llm._reasoning.Platform",
            return_value=mock_platform,
        ):
            stage = ReasoningStage(model_id=ModelID.YOLO_V8, manager=mock_manager)

        assert stage._handle is mock_handle
        mock_platform.load_tflite.assert_called_once_with(ComputeUnit.CPU, fake_path)

    def test_reasoning_stage_with_dlc_model_id(self) -> None:
        """ReasoningStage with a .dlc path calls platform.load_dlc."""
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from moment_to_action.models import ModelID

        fake_path = Path("/fake/model.dlc")
        mock_manager = MagicMock()
        mock_manager.get_path.return_value = fake_path

        mock_platform = MagicMock()
        mock_handle = MagicMock()
        mock_platform.load_dlc.return_value = mock_handle

        with patch(
            "moment_to_action.stages.llm._reasoning.Platform",
            return_value=mock_platform,
        ):
            stage = ReasoningStage(model_id=ModelID.YOLO_V8, manager=mock_manager)

        assert stage._handle is mock_handle
        mock_platform.load_dlc.assert_called_once_with(ComputeUnit.CPU, fake_path)

    def test_reasoning_stage_unknown_extension_raises(self) -> None:
        """ReasoningStage raises ValueError for unknown model file extension."""
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from moment_to_action.models import ModelID

        fake_path = Path("/fake/model.pb")
        mock_manager = MagicMock()
        mock_manager.get_path.return_value = fake_path

        with patch("moment_to_action.stages.llm._reasoning.Platform"):
            with pytest.raises(ValueError, match="Unknown model format"):
                ReasoningStage(model_id=ModelID.YOLO_V8, manager=mock_manager)
