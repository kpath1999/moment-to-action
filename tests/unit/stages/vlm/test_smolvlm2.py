"""Unit tests for SmolVLM2Stage."""

from __future__ import annotations

import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from moment_to_action.models import ModelID, ModelManager
from moment_to_action.stages.vlm._smolvlm2 import SmolVLM2Stage


@pytest.mark.unit
class TestSmolVLM2StageInit:
    """Initialization tests for SmolVLM2Stage."""

    def test_init_uses_model_manager_for_model_path(self) -> None:
        """Stage init should resolve model directory via ModelManager."""
        manager = mock.MagicMock(spec=ModelManager)
        model_dir = Path("/tmp/smolvlm2")
        manager.get_path.return_value = model_dir

        model = mock.MagicMock()
        model.to.return_value = model

        with (
            mock.patch("moment_to_action.stages.vlm._smolvlm2.AutoProcessor.from_pretrained"),
            mock.patch(
                "moment_to_action.stages.vlm._smolvlm2.AutoModelForImageTextToText.from_pretrained",
                return_value=model,
            ) as mock_model_from_pretrained,
        ):
            SmolVLM2Stage(manager=manager)

        manager.get_path.assert_called_once_with(ModelID.SMOLVLM2_2_2B)
        mock_model_from_pretrained.assert_called_once_with(
            model_dir,
            dtype=torch.float32,
            trust_remote_code=True,
        )

    def test_init_loads_processor_from_resolved_model_directory(self) -> None:
        """Stage init should load processor from manager-resolved directory."""
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
            SmolVLM2Stage(manager=manager)

        mock_processor_from_pretrained.assert_called_once_with(
            model_dir,
            trust_remote_code=True,
        )

    def test_init_requires_manager(self) -> None:
        """SmolVLM2Stage requires manager as a positional arg; backend is no longer a param."""
        import inspect

        sig = inspect.signature(SmolVLM2Stage.__init__)
        params = list(sig.parameters.keys())
        assert "manager" in params
        assert "backend" not in params
        assert sig.parameters["manager"].default is inspect.Parameter.empty


@pytest.mark.unit
class TestToPilRgb:
    """Tests for to_pil_rgb helper (now in utils.video)."""

    def test_converts_bgr_to_rgb_pil(self) -> None:
        """Converts an OpenCV BGR ndarray to a PIL RGB image."""
        from moment_to_action.utils.video import to_pil_rgb

        bgr = np.zeros((100, 200, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255  # Blue channel in BGR

        pil_img = to_pil_rgb(bgr)

        assert pil_img.mode == "RGB"
        assert pil_img.size == (200, 100)
        pixel = pil_img.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        r, g, b = pixel
        assert r == 0
        assert g == 0
        assert b == 255


@pytest.mark.unit
class TestSampleFrames:
    """Tests for sample_frames helper (now in utils.video)."""

    def test_returns_all_when_under_limit(self) -> None:
        """Returns all frames when count is below max_images."""
        from moment_to_action.utils.video import sample_frames

        frames = [np.zeros((10, 10)) for _ in range(3)]
        result = sample_frames(frames, max_images=8)
        assert len(result) == 3
        assert result is frames

    def test_samples_uniformly(self) -> None:
        """Uniformly samples when frame count exceeds max_images."""
        from moment_to_action.utils.video import sample_frames

        frames = [np.full((1, 1), i) for i in range(20)]
        result = sample_frames(frames, max_images=4)
        assert len(result) == 4
        np.testing.assert_array_equal(result[0], frames[0])
        np.testing.assert_array_equal(result[-1], frames[-1])


@pytest.mark.unit
class TestSmolVLM2StageProcess:
    """Tests for SmolVLM2Stage._process."""

    @staticmethod
    def _make_stage() -> SmolVLM2Stage:
        """Build a SmolVLM2Stage with fully mocked internals."""
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
            return SmolVLM2Stage(manager=manager)

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
            stage.process(msg)

    def test_process_returns_classification_message(self) -> None:
        """Successful processing returns a ClassificationMessage."""
        from moment_to_action.messages.video import VideoClipMessage
        from moment_to_action.messages.vlm import ClassificationMessage

        stage = self._make_stage()

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
        result = stage.process(msg)

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
        assert stage.process(msg) is None

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
        assert stage.process(msg) is None


@pytest.mark.unit
class TestSmolVLM2CleanGeneration:
    """Tests for SmolVLM2Stage._clean_generation."""

    def test_strips_assistant_marker(self) -> None:
        """Strips text before and including the last Assistant: marker."""
        result = SmolVLM2Stage._clean_generation("some text Assistant: The answer. ")
        assert result == "The answer."

    def test_no_marker_returns_stripped(self) -> None:
        """Without marker, returns stripped text."""
        result = SmolVLM2Stage._clean_generation("  Just a sentence.  ")
        assert result == "Just a sentence."

    def test_empty_string(self) -> None:
        """Empty input returns empty string."""
        result = SmolVLM2Stage._clean_generation("")
        assert result == ""


@pytest.mark.unit
class TestResolveDeviceDtype:
    """Tests for _resolve_torch_device_dtype."""

    def test_explicit_device_used_directly(self) -> None:
        """Non-'auto' requested device is used as-is."""
        from moment_to_action.stages.vlm._smolvlm2 import _resolve_torch_device_dtype

        device, _ = _resolve_torch_device_dtype("cpu")
        assert device.type == "cpu"

    def test_auto_cuda_available(self) -> None:
        """'auto' selects cuda when cuda is available."""
        from moment_to_action.stages.vlm._smolvlm2 import _resolve_torch_device_dtype

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
        ):
            device, dtype = _resolve_torch_device_dtype("auto")
        assert device.type == "cuda"
        assert dtype == torch.bfloat16

    def test_auto_mps_available(self) -> None:
        """'auto' selects mps when cuda is absent but mps is available."""
        from moment_to_action.stages.vlm._smolvlm2 import _resolve_torch_device_dtype

        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=True),
        ):
            device, dtype = _resolve_torch_device_dtype("auto")
        assert device.type == "mps"
        assert dtype == torch.float16
