from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from moment_to_action.benchmark import SmolVLM2Benchmark
from moment_to_action.benchmark._benchmarks._smolvlm2 import (
    _sample_video_frames,
    _SmolVLM2Handle,
)
from moment_to_action.benchmark._datasets._msrvtt_dataset import MsrvttItem
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
def test_smolvlm2_load_uses_torch_policy() -> None:
    benchmark = SmolVLM2Benchmark()
    backend = mock.MagicMock()
    policy = mock.MagicMock(device=torch.device("cpu"), dtype=torch.float32)
    backend.resolve_torch_policy.return_value = policy

    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/smolvlm2")

    processor = mock.MagicMock()
    processor.apply_chat_template.return_value = {
        "input_ids": torch.zeros((1, 4), dtype=torch.long),
    }

    model = mock.MagicMock()
    model.to.return_value = model
    model.device = torch.device("cpu")

    with (
        mock.patch(
            "moment_to_action.benchmark._benchmarks._smolvlm2.AutoProcessor.from_pretrained",
            return_value=processor,
        ) as mock_processor,
        mock.patch(
            "moment_to_action.benchmark._benchmarks._smolvlm2.AutoModelForImageTextToText.from_pretrained",
            return_value=model,
        ) as mock_model,
    ):
        handle = benchmark._load_model(backend=backend, manager=manager)

    manager.get_path.assert_called_once_with(ModelID.SMOLVLM2_2_2B)
    mock_processor.assert_called_once()
    mock_model.assert_called_once()

    inputs = benchmark._make_dummy_input(handle, batch_size=1)
    assert "input_ids" in inputs  # type: ignore[operator]


@pytest.mark.unit
def test_smolvlm2_evaluate_accuracy_exact_match() -> None:
    dataset = mock.MagicMock()
    dataset.items.return_value = [
        MsrvttItem(video_path=Path("/tmp/a.mp4"), question="q1", answer="person running"),
        MsrvttItem(video_path=Path("/tmp/b.mp4"), question="q2", answer="red car"),
    ]
    benchmark = SmolVLM2Benchmark(msrvtt_dataset=dataset)

    with (
        mock.patch.object(SmolVLM2Benchmark, "_cast_handle", return_value=mock.MagicMock()),
        mock.patch(
            "moment_to_action.benchmark._benchmarks._smolvlm2._sample_video_frames",
            return_value=[mock.MagicMock()],
        ),
        mock.patch.object(
            SmolVLM2Benchmark,
            "_generate_answer",
            side_effect=["person running", "different answer"],
        ),
    ):
        accuracy = benchmark._evaluate_accuracy(
            handle=object(),
            backend=mock.MagicMock(),
            manager=mock.MagicMock(),
        )

    assert accuracy == pytest.approx(0.5)


@pytest.mark.unit
def test_smolvlm2_run_inference_type_error() -> None:
    benchmark = SmolVLM2Benchmark()
    with pytest.raises(TypeError, match="Invalid SmolVLM2 benchmark handle"):
        benchmark._run_inference(object(), mock.MagicMock(), mock.MagicMock())


@pytest.mark.unit
def test_smolvlm2_evaluate_accuracy_none_without_dataset() -> None:
    benchmark = SmolVLM2Benchmark(msrvtt_dataset=None)
    result = benchmark._evaluate_accuracy(object(), mock.MagicMock(), mock.MagicMock())
    assert result is None


@pytest.mark.unit
def test_smolvlm2_generate_answer_empty_decode() -> None:
    processor = mock.MagicMock()
    processor.apply_chat_template.return_value = {
        "input_ids": torch.zeros((1, 2), dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 224, 224)),
    }
    processor.batch_decode.return_value = []

    model = mock.MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.zeros((1, 2), dtype=torch.long)

    handle = _SmolVLM2Handle(model=model, processor=processor)
    result = SmolVLM2Benchmark._generate_answer(
        handle,
        "what is happening?",
        [mock.MagicMock()],
    )
    assert result == ""


@pytest.mark.unit
def test_smolvlm2_cast_handle_type_error() -> None:
    with pytest.raises(TypeError, match="Invalid SmolVLM2 benchmark handle"):
        SmolVLM2Benchmark._cast_handle(object())


@pytest.mark.unit
def test_smolvlm2_run_inference_valid_inputs() -> None:
    model = mock.MagicMock()
    model.device = torch.device("cpu")
    processor = mock.MagicMock()
    handle = _SmolVLM2Handle(model=model, processor=processor)

    SmolVLM2Benchmark()._run_inference(
        handle,
        {"input_ids": torch.zeros((1, 2), dtype=torch.long)},
        mock.MagicMock(),
    )
    model.generate.assert_called_once()


@pytest.mark.unit
def test_smolvlm2_run_inference_requires_mapping_for_valid_handle() -> None:
    model = mock.MagicMock()
    model.device = torch.device("cpu")
    handle = _SmolVLM2Handle(model=model, processor=mock.MagicMock())

    with pytest.raises(TypeError, match="expects mapping"):
        SmolVLM2Benchmark()._run_inference(handle, [1, 2], mock.MagicMock())


@pytest.mark.unit
def test_smolvlm2_generate_answer_returns_stripped_text() -> None:
    processor = mock.MagicMock()
    processor.apply_chat_template.return_value = {
        "input_ids": torch.zeros((1, 1), dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
    }
    processor.batch_decode.return_value = ["  hello world  "]

    model = mock.MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.zeros((1, 2), dtype=torch.long)

    handle = _SmolVLM2Handle(model=model, processor=processor)
    result = SmolVLM2Benchmark._generate_answer(handle, "q", [mock.MagicMock()])
    assert result == "hello world"


@pytest.mark.unit
def test_smolvlm2_evaluate_accuracy_none_when_no_frames() -> None:
    dataset = mock.MagicMock()
    dataset.items.return_value = [
        MsrvttItem(video_path=Path("/tmp/a.mp4"), question="q1", answer="a1"),
    ]
    benchmark = SmolVLM2Benchmark(msrvtt_dataset=dataset)

    with (
        mock.patch.object(SmolVLM2Benchmark, "_cast_handle", return_value=mock.MagicMock()),
        mock.patch(
            "moment_to_action.benchmark._benchmarks._smolvlm2._sample_video_frames",
            return_value=[],
        ),
    ):
        result = benchmark._evaluate_accuracy(object(), mock.MagicMock(), mock.MagicMock())
    assert result is None


@pytest.mark.unit
def test_sample_video_frames_handles_unopened_capture() -> None:
    cap = mock.MagicMock()
    cap.isOpened.return_value = False
    with mock.patch("cv2.VideoCapture", return_value=cap):
        frames = _sample_video_frames(Path("/tmp/video.mp4"), max_frames=2)
    assert frames == []


@pytest.mark.unit
def test_sample_video_frames_reads_until_end() -> None:
    cap = mock.MagicMock()
    cap.isOpened.return_value = True
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    cap.read.side_effect = [(True, frame), (False, None)]

    with mock.patch("cv2.VideoCapture", return_value=cap):
        frames = _sample_video_frames(Path("/tmp/video.mp4"), max_frames=3)

    assert len(frames) == 1
    cap.release.assert_called_once()
