"""Unit tests for LlamaServerStage."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from moment_to_action.config import AppConfig
from moment_to_action.messages import DetectionMessage
from moment_to_action.messages.llm import ReasoningMessage
from moment_to_action.messages.sensor import RawFrameMessage
from moment_to_action.models._model_info import ModelID
from moment_to_action.models.image.detection._types import BoundingBox, Detection
from moment_to_action.stages.llm._llama_server import LlamaServerStage


def _det(
    label: str,
    confidence: float,
    x1: float = 0.0,
    y1: float = 0.0,
    x2: float = 100.0,
    y2: float = 100.0,
) -> Detection:
    """Build a Detection fixture."""
    return Detection(
        label=label, confidence=confidence, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    )


def _config(server_path: Path | None = Path("/bin/llama-server"), port: int = 8080) -> AppConfig:
    """Build an AppConfig fixture."""
    return AppConfig(llama_server_path=server_path, llama_server_port=port)


def _make_stage(
    config: AppConfig | None = None,
    response_text: str = "A person is nearby.",
    system_prompt: str = "Be concise.",
    max_tokens: int = 64,
) -> LlamaServerStage:
    """Construct a LlamaServerStage with a mocked model."""
    if config is None:
        config = _config()

    mock_model = MagicMock()
    mock_model.prepare.side_effect = lambda p: {"messages": [], "max_tokens": max_tokens}
    mock_model.run.return_value = response_text
    mock_model.post_proc.side_effect = lambda r: [r]

    mock_manager = MagicMock()
    mock_manager.get_model.return_value = mock_model

    with patch("moment_to_action.stages.llm._llama_server.Platform"):
        return LlamaServerStage(
            mock_manager,
            config,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )


@pytest.mark.unit
class TestLlamaServerStageInit:
    """Tests for LlamaServerStage.__init__."""

    def test_raises_when_llama_server_path_is_none(self) -> None:
        """__init__ raises RuntimeError when llama_server_path is None."""
        config = _config(server_path=None)
        with pytest.raises(RuntimeError, match="llama_server_path"):
            LlamaServerStage(MagicMock(), config)

    def test_calls_get_model_with_server_kwargs(self) -> None:
        """__init__ forwards server_path, port, system_prompt, max_tokens to get_model."""
        config = _config(server_path=Path("/custom/llama"), port=9999)
        mock_model = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_model.return_value = mock_model

        with patch("moment_to_action.stages.llm._llama_server.Platform"):
            LlamaServerStage(
                mock_manager,
                config,
                model_id=ModelID.QWEN2_1_5B_INSTRUCT,
                system_prompt="sys",
                max_tokens=32,
            )

        mock_manager.get_model.assert_called_once_with(
            ModelID.QWEN2_1_5B_INSTRUCT,
            server_path=Path("/custom/llama"),
            port=9999,
            system_prompt="sys",
            max_tokens=32,
        )

    def test_calls_model_load(self) -> None:
        """__init__ calls model.load() with a Platform and ComputeUnit."""
        from moment_to_action.hardware._types import ComputeUnit

        mock_model = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_model.return_value = mock_model

        mock_platform = MagicMock()
        with patch(
            "moment_to_action.stages.llm._llama_server.Platform",
            return_value=mock_platform,
        ):
            LlamaServerStage(mock_manager, _config())

        mock_model.load.assert_called_once_with(mock_platform, ComputeUnit.CPU)

    def test_stage_name(self) -> None:
        """Name property returns 'LlamaServerStage'."""
        stage = _make_stage()
        assert stage.name == "LlamaServerStage"


@pytest.mark.unit
class TestLlamaServerStageProcess:
    """Tests for LlamaServerStage._process."""

    @pytest.fixture
    def detection_msg(self) -> DetectionMessage:
        """DetectionMessage with sample detections."""
        return DetectionMessage(
            detections=[
                _det("person", 0.95, 10.0, 20.0, 200.0, 400.0),
                _det("chair", 0.70, 300.0, 100.0, 500.0, 300.0),
            ],
            timestamp=time.time(),
        )

    def test_process_returns_reasoning_message(self, detection_msg: DetectionMessage) -> None:
        """_process returns a ReasoningMessage."""
        stage = _make_stage()
        result = stage.process(detection_msg)
        assert isinstance(result, ReasoningMessage)

    def test_process_response_text(self, detection_msg: DetectionMessage) -> None:
        """_process populates response with the model output."""
        stage = _make_stage(response_text="Scene: person detected.")
        result = stage.process(detection_msg)
        assert isinstance(result, ReasoningMessage)
        assert result.response == "Scene: person detected."

    def test_process_prompt_in_result(self, detection_msg: DetectionMessage) -> None:
        """_process echoes the built prompt in the result."""
        stage = _make_stage()
        result = stage.process(detection_msg)
        assert isinstance(result, ReasoningMessage)
        assert "person" in result.prompt
        assert "Detections:" in result.prompt

    def test_process_preserves_timestamp(self, detection_msg: DetectionMessage) -> None:
        """_process preserves the input message timestamp."""
        stage = _make_stage()
        result = stage.process(detection_msg)
        assert isinstance(result, ReasoningMessage)
        assert result.timestamp == detection_msg.timestamp

    def test_process_raises_on_wrong_message_type(self) -> None:
        """_process raises TypeError when msg is not a DetectionMessage."""
        stage = _make_stage()
        wrong_msg = RawFrameMessage(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=time.time(),
            width=640,
            height=480,
        )
        with pytest.raises(TypeError, match="DetectionMessage"):
            stage.process(wrong_msg)

    def test_process_uses_metrics_span(self, detection_msg: DetectionMessage) -> None:
        """_process wraps inference in a MODEL_INFERENCE metrics span."""
        from moment_to_action.metrics import MetricsCollector

        stage = _make_stage()
        metrics = MetricsCollector(session_id="test_span")
        with metrics.start_trace():
            result = stage.process(detection_msg, metrics=metrics)
        assert result is not None
        assert result.latency_ms >= 0.0

    def test_process_empty_detections(self) -> None:
        """_process handles DetectionMessage with no detections."""
        stage = _make_stage()
        msg = DetectionMessage(detections=[], timestamp=time.time())
        result = stage.process(msg)
        assert isinstance(result, ReasoningMessage)


@pytest.mark.unit
class TestLlamaServerStageBuildPrompt:
    """Tests for LlamaServerStage._build_prompt."""

    @pytest.fixture
    def stage(self) -> LlamaServerStage:
        """LlamaServerStage instance."""
        return _make_stage()

    def test_build_prompt_includes_detections_header(self, stage: LlamaServerStage) -> None:
        """_build_prompt includes 'Detections:' header."""
        msg = DetectionMessage(
            detections=[_det("cat", 0.9)],
            timestamp=time.time(),
        )
        prompt = stage._build_prompt(msg)
        assert "Detections:" in prompt

    def test_build_prompt_includes_label(self, stage: LlamaServerStage) -> None:
        """_build_prompt includes detection labels."""
        msg = DetectionMessage(
            detections=[_det("bicycle", 0.8)],
            timestamp=time.time(),
        )
        assert "bicycle" in stage._build_prompt(msg)

    def test_build_prompt_includes_confidence(self, stage: LlamaServerStage) -> None:
        """_build_prompt includes confidence scores."""
        msg = DetectionMessage(
            detections=[_det("dog", 0.77)],
            timestamp=time.time(),
        )
        assert "0.77" in stage._build_prompt(msg)

    def test_build_prompt_includes_bbox(self, stage: LlamaServerStage) -> None:
        """_build_prompt includes bounding box coordinates."""
        msg = DetectionMessage(
            detections=[_det("car", 0.9, x1=10.0, y1=20.0, x2=300.0, y2=400.0)],
            timestamp=time.time(),
        )
        prompt = stage._build_prompt(msg)
        assert "10" in prompt
        assert "20" in prompt

    def test_build_prompt_top5_only(self, stage: LlamaServerStage) -> None:
        """_build_prompt selects at most 5 detections."""
        dets = [_det(f"obj{i}", 1.0 - i * 0.1) for i in range(8)]
        msg = DetectionMessage(detections=dets, timestamp=time.time())
        prompt = stage._build_prompt(msg)
        detection_lines = [l for l in prompt.split("\n") if l.strip().startswith("-")]
        assert len(detection_lines) == 5

    def test_build_prompt_ordered_by_confidence(self, stage: LlamaServerStage) -> None:
        """_build_prompt lists detections highest-confidence first."""
        dets = [_det("low", 0.2), _det("high", 0.9), _det("mid", 0.5)]
        msg = DetectionMessage(detections=dets, timestamp=time.time())
        prompt = stage._build_prompt(msg)
        idx_high = prompt.index("0.90")
        idx_mid = prompt.index("0.50")
        idx_low = prompt.index("0.20")
        assert idx_high < idx_mid < idx_low

    def test_build_prompt_ends_with_question(self, stage: LlamaServerStage) -> None:
        """_build_prompt ends with the scene question."""
        msg = DetectionMessage(detections=[], timestamp=time.time())
        prompt = stage._build_prompt(msg)
        assert "What is happening in this scene?" in prompt


@pytest.mark.unit
class TestLlamaServerStageClose:
    """Tests for LlamaServerStage.close()."""

    def _make_stage_with_mock(self) -> tuple[LlamaServerStage, MagicMock]:
        """Construct stage and return both the stage and the underlying mock model."""
        mock_model = MagicMock()
        mock_model.prepare.side_effect = lambda p: {"messages": [], "max_tokens": 64}
        mock_model.run.return_value = "text"
        mock_model.post_proc.side_effect = lambda r: [r]

        mock_manager = MagicMock()
        mock_manager.get_model.return_value = mock_model

        with patch("moment_to_action.stages.llm._llama_server.Platform"):
            stage = LlamaServerStage(mock_manager, _config())

        return stage, mock_model

    def test_close_calls_model_unload(self) -> None:
        """close() calls model.unload()."""
        stage, mock_model = self._make_stage_with_mock()
        stage.close()
        mock_model.unload.assert_called_once()

    def test_close_is_idempotent(self) -> None:
        """Second close() is a no-op (unload not called twice)."""
        stage, mock_model = self._make_stage_with_mock()
        stage.close()
        stage.close()
        mock_model.unload.assert_called_once()
