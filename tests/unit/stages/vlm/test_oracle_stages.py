from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from moment_to_action.messages import ClassificationMessage, FrameTensorMessage
from moment_to_action.models import ModelManager
from moment_to_action.stages.vlm._oracle_dino import OracleGroundingDinoStage
from moment_to_action.stages.vlm._oracle_siglip import OracleSigLipStage


def _mock_metrics() -> mock.MagicMock:
    metrics = mock.MagicMock()
    metrics.start_span.return_value = nullcontext()
    return metrics


@pytest.mark.unit
def test_oracle_dino_init_uses_resolved_device() -> None:
    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/dino")

    policy = mock.MagicMock(device=torch.device("cpu"))
    model = mock.MagicMock()
    model.to.return_value = model

    with (
        mock.patch(
            "moment_to_action.stages.vlm._oracle_dino.resolve_torch_execution_policy",
            return_value=policy,
        ),
        mock.patch(
            "moment_to_action.stages.vlm._oracle_dino.AutoProcessor.from_pretrained",
            return_value=mock.MagicMock(),
        ),
        mock.patch(
            "moment_to_action.stages.vlm._oracle_dino.AutoModelForZeroShotObjectDetection.from_pretrained",
            return_value=model,
        ),
    ):
        stage = OracleGroundingDinoStage(["person"], manager, torch_device="auto")

    assert str(stage._device) == "cpu"


@pytest.mark.unit
def test_oracle_dino_process_non_frame_returns_none() -> None:
    stage = OracleGroundingDinoStage.__new__(OracleGroundingDinoStage)
    msg = ClassificationMessage(label="x", confidence=0.1, all_scores={"x": 0.1}, timestamp=0.0)
    result = stage._process(msg=msg, metrics=_mock_metrics())
    assert result is None


@pytest.mark.unit
def test_oracle_dino_process_happy_path() -> None:
    stage = OracleGroundingDinoStage.__new__(OracleGroundingDinoStage)
    stage._device = torch.device("cpu")
    stage._text_queries = ["person"]

    inputs = {"input_ids": torch.tensor([[1, 2]], dtype=torch.int64)}

    class _Inputs(dict):
        def to(self, device: object) -> _Inputs:
            del device
            return self

        @property
        def input_ids(self) -> torch.Tensor:
            return self["input_ids"]

    stage._processor = mock.MagicMock()
    stage._processor.side_effect = None
    stage._processor.return_value = _Inputs(inputs)
    stage._processor.post_process_grounded_object_detection.return_value = [
        {
            "boxes": [torch.tensor([1.0, 2.0, 3.0, 4.0])],
            "scores": [torch.tensor(0.9)],
            "labels": ["person"],
        }
    ]
    stage._model = mock.MagicMock(return_value=mock.MagicMock())

    frame = FrameTensorMessage(
        tensor=np.zeros((1, 3, 8, 8), dtype=np.float32),
        original_size=(8, 8),
        timestamp=1.23,
    )
    result = stage._process(frame, _mock_metrics())
    assert result is not None
    assert len(result.boxes) == 1
    assert result.boxes[0].label == "person"


@pytest.mark.unit
def test_oracle_siglip_init_uses_resolved_device() -> None:
    manager = mock.MagicMock(spec=ModelManager)
    manager.get_path.return_value = Path("/tmp/siglip")
    policy = mock.MagicMock(device=torch.device("cpu"))
    model = mock.MagicMock()
    model.to.return_value = model

    with (
        mock.patch(
            "moment_to_action.stages.vlm._oracle_siglip.resolve_torch_execution_policy",
            return_value=policy,
        ),
        mock.patch(
            "moment_to_action.stages.vlm._oracle_siglip.AutoProcessor.from_pretrained",
            return_value=mock.MagicMock(),
        ),
        mock.patch(
            "moment_to_action.stages.vlm._oracle_siglip.AutoModel.from_pretrained",
            return_value=model,
        ),
    ):
        stage = OracleSigLipStage(["a person"], manager, torch_device="auto")

    assert str(stage._device) == "cpu"


@pytest.mark.unit
def test_oracle_siglip_process_non_frame_returns_none() -> None:
    stage = OracleSigLipStage.__new__(OracleSigLipStage)
    msg = ClassificationMessage(label="x", confidence=0.1, all_scores={"x": 0.1}, timestamp=0.0)
    result = stage._process(msg=msg, metrics=_mock_metrics())
    assert result is None


@pytest.mark.unit
def test_oracle_siglip_process_happy_path() -> None:
    stage = OracleSigLipStage.__new__(OracleSigLipStage)
    stage._device = torch.device("cpu")
    stage._text_prompts = ["a person", "a car"]

    class _Inputs(dict):
        def to(self, device: object) -> _Inputs:
            del device
            return self

    stage._processor = mock.MagicMock(return_value=_Inputs({"input_ids": torch.tensor([[1]])}))
    outputs = mock.MagicMock()
    outputs.logits_per_image = torch.tensor([[5.0, -1.0]], dtype=torch.float32)
    stage._model = mock.MagicMock(return_value=outputs)

    frame = FrameTensorMessage(
        tensor=np.zeros((1, 3, 8, 8), dtype=np.float32),
        original_size=(8, 8),
        timestamp=2.34,
    )
    result = stage._process(frame, _mock_metrics())
    assert result is not None
    assert result.label == "a person"
