"""Unit tests for TorchModel."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from moment_to_action.hardware._loaded_models._torch import TorchModel
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType


@pytest.mark.unit
class TestTorchModel:
    """Tests for TorchModel properties and inference."""

    def _make_model(self, unit: ComputeUnit = ComputeUnit.CPU) -> TorchModel:
        """Return a TorchModel with a callable mock."""
        mock_module = MagicMock(return_value="output")
        return TorchModel(unit=unit, model=mock_module)

    def test_unit_property(self) -> None:
        """Unit returns the value passed at construction."""
        assert self._make_model(ComputeUnit.CPU).unit == ComputeUnit.CPU
        assert self._make_model(ComputeUnit.GPU).unit == ComputeUnit.GPU

    def test_dtype_property(self) -> None:
        """Dtype is always FP32."""
        assert self._make_model().dtype == DataType.FP32

    def test_model_type_property(self) -> None:
        """model_type is always TORCH."""
        assert self._make_model().model_type == ModelType.TORCH

    def test_run_calls_model(self) -> None:
        """run() calls the underlying model with inputs and returns result."""
        model = self._make_model()
        result = model.run("some_input")
        assert result == "output"

    def test_run_when_unloaded_raises(self) -> None:
        """run() raises RuntimeError after unload()."""
        model = self._make_model()
        model.unload()
        with pytest.raises(RuntimeError, match="unloaded"):
            model.run("input")

    def test_unload_clears_model(self) -> None:
        """unload() clears the model handle and sets _unloaded."""
        model = self._make_model()
        model.unload()
        assert model._model is None
        assert model._unloaded is True

    def test_unload_idempotent(self) -> None:
        """Calling unload() twice is safe."""
        model = self._make_model()
        model.unload()
        model.unload()
        assert model._unloaded is True
