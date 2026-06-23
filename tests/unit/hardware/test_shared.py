"""Unit tests for hardware._platforms._shared helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from moment_to_action.hardware._platforms._shared import (
    _load_litert_interpreter,
    _tflite_set_inputs,
)


@pytest.mark.unit
class TestLoadLiteRTInterpreter:
    """Tests for _load_litert_interpreter."""

    def test_returns_allocated_interpreter(self) -> None:
        """Returns the interpreter after allocate_tensors is called."""
        mock_interp = MagicMock()
        with patch(
            "ai_edge_litert.interpreter.Interpreter",
            return_value=mock_interp,
        ) as mock_cls:
            result = _load_litert_interpreter("/tmp/model.tflite")

        mock_cls.assert_called_once_with(model_path="/tmp/model.tflite", experimental_delegates=[])
        mock_interp.allocate_tensors.assert_called_once()
        assert result is mock_interp

    def test_passes_delegates_to_interpreter(self) -> None:
        """Delegates list is forwarded to the Interpreter constructor."""
        mock_delegate = MagicMock()
        mock_interp = MagicMock()
        with patch(
            "ai_edge_litert.interpreter.Interpreter",
            return_value=mock_interp,
        ) as mock_cls:
            _load_litert_interpreter("/tmp/model.tflite", delegates=[mock_delegate])

        mock_cls.assert_called_once_with(
            model_path="/tmp/model.tflite", experimental_delegates=[mock_delegate]
        )

    def test_none_delegates_defaults_to_empty_list(self) -> None:
        """delegates=None is treated the same as an empty list."""
        mock_interp = MagicMock()
        with patch(
            "ai_edge_litert.interpreter.Interpreter",
            return_value=mock_interp,
        ) as mock_cls:
            _load_litert_interpreter("/tmp/model.tflite", delegates=None)

        _, kwargs = mock_cls.call_args
        assert kwargs["experimental_delegates"] == []


@pytest.mark.unit
class TestTfliteSetInputs:
    """Tests for _tflite_set_inputs."""

    def _make_interp(self, input_details: list) -> MagicMock:
        """Build a mock interpreter with given input_details."""
        interp = MagicMock()
        interp.get_input_details.return_value = input_details
        return interp

    def test_single_ndarray_uses_index_zero(self) -> None:
        """Single ndarray is set at input_details[0]['index']."""
        detail = {"index": 5, "name": "input", "dtype": np.float32}
        interp = self._make_interp([detail])
        arr = np.zeros((1, 3, 224, 224), dtype=np.float32)

        _tflite_set_inputs(interp, arr)

        interp.set_tensor.assert_called_once_with(5, arr)

    def test_dict_input_sets_each_tensor_by_name(self) -> None:
        """Dict inputs are set by matching name → index."""
        details = [
            {"index": 0, "name": "img", "dtype": np.float32},
            {"index": 1, "name": "mask", "dtype": np.float32},
        ]
        interp = self._make_interp(details)
        img = np.zeros((1, 3, 224, 224), dtype=np.float32)
        mask = np.zeros((1, 1, 224, 224), dtype=np.float32)

        _tflite_set_inputs(interp, {"img": img, "mask": mask})

        calls = {call[0][0]: call[0][1] for call in interp.set_tensor.call_args_list}
        assert calls[0] is img
        assert calls[1] is mask

    def test_missing_input_name_raises_key_error(self) -> None:
        """KeyError raised when dict key not in model input names."""
        detail = {"index": 0, "name": "img", "dtype": np.float32}
        interp = self._make_interp([detail])
        tensor = np.zeros((1,), dtype=np.float32)

        with pytest.raises(KeyError, match="wrong"):
            _tflite_set_inputs(interp, {"wrong": tensor})

    def test_dtype_mismatch_raises_type_error(self) -> None:
        """TypeError raised when tensor dtype does not match model's expected dtype."""
        detail = {"index": 0, "name": "img", "dtype": np.float32}
        interp = self._make_interp([detail])
        tensor = np.zeros((1,), dtype=np.int32)

        with pytest.raises(TypeError, match="dtype mismatch"):
            _tflite_set_inputs(interp, {"img": tensor})
