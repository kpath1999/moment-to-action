"""Unit tests for SmolVLM2Stage."""

from __future__ import annotations

import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from moment_to_action.models import ModelID, ModelManager
from moment_to_action.stages.vlm._smolvlm2 import (
    SmolVLM2Stage,
    _sample_frames,
    _to_pil_rgb,
    _torch_dtype_from_name,
)


@pytest.mark.unit
class TestSmolVLM2StageInit:
    """Initialization tests for SmolVLM2Stage."""

    def test_init_uses_model_manager_for_model_path(self) -> None:
        """Stage init should resolve model directory via ModelManager."""
        backend = mock.MagicMock()
        backend.resolve_torch_policy.return_value = mock.MagicMock(device="cpu", dtype="float32")

        manager = mock.MagicMock(spec=ModelManager)
        model_dir = Path("/tmp/smolvlm2")
        manager.get_path.return_value = model_dir

        model = mock.MagicMock()
        model.to.return_value = model

        with (
            mock.patch(
                "moment_to_action.stages.vlm._smolvlm2.AutoProcessor.from_pretrained"
            ) as _mock_processor,
            mock.patch(
                "moment_to_action.stages.vlm._smolvlm2.AutoModelForImageTextToText.from_pretrained",
                return_value=model,
            ) as mock_model_from_pretrained,
        ):
            SmolVLM2Stage(backend=backend, manager=manager)

        manager.get_path.assert_called_once_with(ModelID.SMOLVLM2_2_2B)
        mock_model_from_pretrained.assert_called_once_with(
            model_dir,
            dtype=torch.float32,
            trust_remote_code=True,
        )

    def test_init_loads_processor_from_resolved_model_directory(self) -> None:
        """Stage init should load processor from manager-resolved directory."""
        backend = mock.MagicMock()
        backend.resolve_torch_policy.return_value = mock.MagicMock(device="cpu", dtype="bfloat16")

        manager = mock.MagicMock(spec=ModelManager)
        model_dir = Path("/tmp/smolvlm2")
        manager.get_path.return_value = model_dir

        model = mock.MagicMock()
        model.to.return_value = model

        with (
            mock.patch(
                "moment_to_action.stages.vlm._smolvlm2.AutoProcessor.from_pretrained"
            ) as mock_processor_from_pretrained,
            mock.patch(
                "moment_to_action.stages.vlm._smolvlm2.AutoModelForImageTextToText.from_pretrained",
                return_value=model,
            ),
        ):
            SmolVLM2Stage(backend=backend, manager=manager)

        mock_processor_from_pretrained.assert_called_once_with(
            model_dir,
            trust_remote_code=True,
        )


@pytest.mark.unit
class TestTorchDtypeFromName:
    """Tests for _torch_dtype_from_name helper."""

    def test_bfloat16(self) -> None:
        """Converts 'bfloat16' to torch.bfloat16."""
        assert _torch_dtype_from_name("bfloat16") is torch.bfloat16

    def test_float16(self) -> None:
        """Converts 'float16' to torch.float16."""
        assert _torch_dtype_from_name("float16") is torch.float16

    def test_float32_default(self) -> None:
        """Unrecognised names fall back to torch.float32."""
        assert _torch_dtype_from_name("float32") is torch.float32
        assert _torch_dtype_from_name("anything") is torch.float32


@pytest.mark.unit
class TestToPilRgb:
    """Tests for _to_pil_rgb helper."""

    def test_converts_bgr_to_rgb_pil(self) -> None:
        """Converts an OpenCV BGR ndarray to a PIL RGB image."""
        bgr = np.zeros((100, 200, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # Blue channel

        pil_img = _to_pil_rgb(bgr)

        assert pil_img.mode == "RGB"
        assert pil_img.size == (200, 100)
        # After BGR->RGB, blue channel (index 0 of BGR) becomes index 2 of RGB
        pixel = pil_img.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        r, g, b = pixel
        assert r == 0
        assert g == 0
        assert b == 255


@pytest.mark.unit
class TestSampleFrames:
    """Tests for _sample_frames helper."""

    def test_returns_all_when_under_limit(self) -> None:
        """Returns all frames when count is below max_images."""
        frames = [np.zeros((10, 10)) for _ in range(3)]
        result = _sample_frames(frames, max_images=8)
        assert len(result) == 3
        assert result is frames  # no copy

    def test_samples_uniformly(self) -> None:
        """Uniformly samples when frame count exceeds max_images."""
        frames = [np.full((1, 1), i) for i in range(20)]
        result = _sample_frames(frames, max_images=4)
        assert len(result) == 4
        # Should include first and last
        np.testing.assert_array_equal(result[0], frames[0])
        np.testing.assert_array_equal(result[-1], frames[-1])


@pytest.mark.unit
class TestSmolVLM2StageProcess:
    """Tests for SmolVLM2Stage._process."""

    @staticmethod
    def _make_stage() -> SmolVLM2Stage:
        """Build a SmolVLM2Stage with fully mocked internals."""
        backend = mock.MagicMock()
        backend.resolve_torch_policy.return_value = mock.MagicMock(device="cpu", dtype="float32")

        manager = mock.MagicMock(spec=ModelManager)
        manager.get_path.return_value = Path("/tmp/smolvlm2")

        model = mock.MagicMock()
        model.to.return_value = model
        model.device = "cpu"

        with (
            mock.patch("moment_to_action.stages.vlm._smolvlm2.AutoProcessor.from_pretrained"),
            mock.patch(
                "moment_to_action.stages.vlm._smolvlm2.AutoModelForImageTextToText.from_pretrained",
                return_value=model,
            ),
        ):
            return SmolVLM2Stage(backend=backend, manager=manager)

    def test_wrong_message_type_raises(self) -> None:
        """Passing non-VideoClipMessage raises TypeError."""
        from moment_to_action.messages.sensor import RawFrameMessage

        stage = self._make_stage()
        msg = RawFrameMessage(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            timestamp=time.time(),
            source="test",
        )
        with pytest.raises(TypeError, match="expects VideoClipMessage"):
            stage._process(msg)

    def test_process_returns_classification_message(self) -> None:
        """Successful processing returns a ClassificationMessage."""
        from moment_to_action.messages.video import VideoClipMessage
        from moment_to_action.messages.vlm import ClassificationMessage

        stage = self._make_stage()

        # Mock the processor and model generate/decode
        stage._processor = mock.MagicMock()
        stage._processor.apply_chat_template.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
        }
        stage._processor.batch_decode.return_value = ["A person is walking."]

        stage._model = mock.MagicMock()
        stage._model.device = "cpu"
        stage._model.generate.return_value = torch.zeros(1, 20, dtype=torch.long)

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]
        msg = VideoClipMessage(timestamp=time.time(), frames=frames, source="test")
        result = stage._process(msg)

        assert isinstance(result, ClassificationMessage)
        assert result.label == "A person is walking."
        assert result.confidence == 1.0

    def test_process_returns_none_on_empty_caption(self) -> None:
        """Returns None when the model generates an empty caption."""
        from moment_to_action.messages.video import VideoClipMessage

        stage = self._make_stage()

        stage._processor = mock.MagicMock()
        stage._processor.apply_chat_template.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
        }
        stage._processor.batch_decode.return_value = [""]

        stage._model = mock.MagicMock()
        stage._model.device = "cpu"
        stage._model.generate.return_value = torch.zeros(1, 20, dtype=torch.long)

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
        msg = VideoClipMessage(timestamp=time.time(), frames=frames, source="test")
        result = stage._process(msg)

        assert result is None

    def test_process_returns_none_on_empty_decoded_list(self) -> None:
        """Returns None when batch_decode returns an empty list."""
        from moment_to_action.messages.video import VideoClipMessage

        stage = self._make_stage()

        stage._processor = mock.MagicMock()
        stage._processor.apply_chat_template.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
        }
        stage._processor.batch_decode.return_value = []

        stage._model = mock.MagicMock()
        stage._model.device = "cpu"
        stage._model.generate.return_value = torch.zeros(1, 20, dtype=torch.long)

        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
        msg = VideoClipMessage(timestamp=time.time(), frames=frames, source="test")
        result = stage._process(msg)

        assert result is None


@pytest.mark.unit
class TestSmolVLM2CleanGeneration:
    """Tests for SmolVLM2Stage._clean_generation."""

    def test_strips_assistant_marker(self) -> None:
        """Strips 'Assistant:' prefix and leading/trailing whitespace."""
        result = SmolVLM2Stage._clean_generation("some text Assistant: The answer is here. ")
        assert result == "The answer is here."

    def test_no_marker_returns_stripped(self) -> None:
        """Without marker, returns stripped text."""
        result = SmolVLM2Stage._clean_generation("  Just a sentence.  ")
        assert result == "Just a sentence."

    def test_empty_string(self) -> None:
        """Empty input returns empty string."""
        result = SmolVLM2Stage._clean_generation("")
        assert result == ""
