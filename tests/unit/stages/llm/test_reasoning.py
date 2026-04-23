"""Unit tests for the PromptMessage-based reasoning stage."""

from __future__ import annotations

import time
from pathlib import Path
from unittest import mock

import pytest

from moment_to_action.messages.llm import ReasoningMessage
from moment_to_action.messages.prompt import PromptMessage
from moment_to_action.messages.video import BoundingBox, DetectionMessage
from moment_to_action.stages.llm._reasoning import (
    _SYSTEMA_PROMPTA,
    _SYSTEMB_PROMPTB,
    ReasoningStage,
)


@pytest.mark.unit
class TestReasoningStage:
    """Tests for the restored PromptMessage-oriented reasoning stage."""

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
        return DetectionMessage(boxes=boxes, timestamp=time.time())

    @pytest.fixture
    def sample_prompt_message(self) -> PromptMessage:
        """Create a prompt message that resembles PromptFormatterStage output."""
        return PromptMessage(
            prompt='{"source": "detection", "detections": [{"label": "person", "confidence": 0.95}]}',
            source_stage="YOLOStage",
            raw_context={"source": "detection"},
            timestamp=time.time(),
        )

    @pytest.fixture
    def mock_llm_runtime(self) -> tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock]:
        """Patch the OpenAI client and LLM metrics endpoint."""
        with (
            mock.patch("moment_to_action.stages.llm._reasoning.OpenAI") as mock_openai,
            mock.patch("moment_to_action.stages.llm._reasoning.httpx.get") as mock_httpx_get,
            mock.patch(
                "moment_to_action.stages.llm._reasoning.psutil.process_iter",
                return_value=[],
            ),
        ):
            mock_client = mock.MagicMock()
            mock_choice = mock.MagicMock()
            mock_choice.message.content = '{"decision":"alert","reason":"test"}'
            mock_response = mock.MagicMock()
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            mock_httpx_get.return_value.json.return_value = [{"n_past": 12, "n_ctx": 512}]
            yield mock_openai, mock_client, mock_httpx_get

    def test_reasoning_stage_initialization_without_model(self) -> None:
        """Default construction leaves backend/handle unset and uses config prompt."""
        stage = ReasoningStage()

        assert stage._backend is None
        assert stage._handle is None
        assert stage._system_prompt == _SYSTEMA_PROMPTA

    def test_reasoning_stage_custom_system_prompt(self) -> None:
        """Custom system prompt overrides the default."""
        custom_prompt = "You are a robot analyzing scenes."
        stage = ReasoningStage(system_prompt=custom_prompt)

        assert stage._system_prompt == custom_prompt

    def test_build_prompt_includes_detection_fields(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """The helper prompt includes the detection list and coordinates."""
        stage = ReasoningStage()

        prompt = stage._build_prompt(sample_detection_message)

        assert prompt.startswith(stage._system_prompt)
        assert "Detections:" in prompt
        assert "person" in prompt
        assert "hand" in prompt
        assert "confidence:" in prompt.lower()
        assert "position:" in prompt.lower()
        assert "What is happening in this scene?" in prompt

    def test_build_prompt_uses_top_five_ordered_detections(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """The helper prompt includes at most the top five detections by confidence."""
        stage = ReasoningStage()

        prompt = stage._build_prompt(sample_detection_message)
        detection_lines = [line for line in prompt.split("\n") if line.strip().startswith("-")]

        assert len(detection_lines) == 5
        assert "0.95" in detection_lines[0]
        assert "0.58" in detection_lines[-1]
        assert "phone" not in prompt

    def test_build_messages_includes_system_prompt_and_user_content(self) -> None:
        """_build_messages preserves the configured system prompt."""
        stage = ReasoningStage(system_prompt="Custom system prompt")

        messages = stage._build_messages("hello world")

        assert messages == [
            {"role": "system", "content": "Custom system prompt"},
            {"role": "user", "content": "hello world"},
        ]

    def test_process_returns_reasoning_message(
        self,
        sample_prompt_message: PromptMessage,
        mock_llm_runtime: tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
    ) -> None:
        """Processing a PromptMessage yields a ReasoningMessage."""
        _, mock_client, _ = mock_llm_runtime
        stage = ReasoningStage()

        result = stage.process(sample_prompt_message)

        assert isinstance(result, ReasoningMessage)
        assert result.response == '{"decision":"alert","reason":"test"}'
        assert result.prompt == sample_prompt_message.prompt
        assert result.timestamp == sample_prompt_message.timestamp
        mock_client.chat.completions.create.assert_called_once()

    def test_process_wraps_prompt_with_reasoning_template(
        self,
        sample_prompt_message: PromptMessage,
        mock_llm_runtime: tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
    ) -> None:
        """The OpenAI request uses the restored template contract."""
        _, mock_client, _ = mock_llm_runtime
        stage = ReasoningStage()

        stage.process(sample_prompt_message)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == [
            {
                "role": "user",
                "content": _SYSTEMB_PROMPTB.replace("{{INPUT_JSON}}", sample_prompt_message.prompt),
            }
        ]

    def test_reasoning_stage_rejects_non_prompt_message(
        self, sample_detection_message: DetectionMessage
    ) -> None:
        """The restored stage expects PromptMessage at process time."""
        stage = ReasoningStage()

        with pytest.raises(TypeError, match="expects PromptMessage"):
            stage.process(sample_detection_message)

    def test_reasoning_stage_name(self) -> None:
        """The stage name falls back to the class name."""
        stage = ReasoningStage()

        assert stage.name == "LLMStage"

    def test_reasoning_stage_latency_stamped(
        self,
        sample_prompt_message: PromptMessage,
        mock_llm_runtime: tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
    ) -> None:
        """Stage.process stamps latency on the result with real metrics."""
        from moment_to_action.metrics import MetricsCollector

        stage = ReasoningStage()
        metrics = MetricsCollector(session_id="test_reasoning_latency")

        with metrics.start_trace():
            result = stage.process(sample_prompt_message, metrics=metrics)

        assert result is not None
        assert isinstance(result, ReasoningMessage)
        assert result.latency_ms >= 0.0

    def test_manager_required_with_model_id(self) -> None:
        """Providing a model ID without a manager is still an error."""
        from moment_to_action.models import ModelID

        with pytest.raises(ValueError, match="Model manager is required"):
            ReasoningStage(model_id=ModelID.YOLO_V8)

        with pytest.raises(ValueError, match="Model manager is required"):
            ReasoningStage(model_id=ModelID.YOLO_V8, manager=None)

    def test_reasoning_stage_with_model_id_mocked(self) -> None:
        """The restored constructor creates a backend but does not load a handle."""
        from moment_to_action.models import ModelID, ModelManager

        fake_path = Path("/fake/model.onnx")
        mock_manager = mock.MagicMock(spec=ModelManager)
        mock_manager.get_path.return_value = fake_path

        mock_backend = mock.MagicMock()

        with mock.patch(
            "moment_to_action.stages.llm._reasoning.ComputeBackend",
            return_value=mock_backend,
        ):
            stage = ReasoningStage(model_id=ModelID.YOLO_V8, manager=mock_manager)

        assert stage._backend is mock_backend
        assert stage._handle is None
        mock_manager.get_path.assert_called_once_with(ModelID.YOLO_V8)
        mock_backend.load_model.assert_not_called()

    def test_llm_metrics_uses_slot_endpoint(
        self, mock_llm_runtime: tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock]
    ) -> None:
        """LLM metrics are read from the slot endpoint in the restored implementation."""
        _, _, mock_httpx_get = mock_llm_runtime
        stage = ReasoningStage()

        metrics = stage._llm_metrics()

        mock_httpx_get.assert_called_once_with("http://localhost:8080/slots")
        assert metrics["kv_cache_used"] == 12
        assert metrics["kv_cache_total"] == 512
