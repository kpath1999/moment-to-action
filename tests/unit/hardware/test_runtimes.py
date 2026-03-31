"""Unit tests for runtime backends (LiteRT, ONNX)."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from moment_to_action.hardware._platforms._runtimes._litert import LiteRTBackend
from moment_to_action.hardware._platforms._runtimes._onnx import ONNXBackend
from moment_to_action.hardware._types import ComputeUnit


@pytest.mark.unit
class TestLiteRTBackend:
    """Test LiteRT backend construction and methods."""

    def test_litert_backend_construction_with_cpu(self) -> None:
        """Test LiteRTBackend construction with CPU unit."""
        backend = LiteRTBackend(compute_unit=ComputeUnit.CPU)
        assert backend is not None
        assert backend.get_supported_unit() == ComputeUnit.CPU

    def test_litert_get_supported_unit_returns_cpu(self) -> None:
        """Test LiteRTBackend.get_supported_unit returns CPU."""
        backend = LiteRTBackend(compute_unit=ComputeUnit.CPU)
        assert backend.get_supported_unit() == ComputeUnit.CPU

    def test_litert_get_supported_unit_returns_specified_unit(self) -> None:
        """Test LiteRTBackend returns the unit specified at construction."""
        backend = LiteRTBackend(compute_unit=ComputeUnit.NPU)
        assert backend.get_supported_unit() == ComputeUnit.NPU

    @patch("moment_to_action.hardware._platforms._runtimes._litert._Interpreter")
    def test_litert_load_model_mocked(self, mock_interpreter_class: Mock) -> None:
        """Test LiteRTBackend.load_model with mocked interpreter."""
        mock_interp = MagicMock()
        mock_interpreter_class.return_value = mock_interp

        backend = LiteRTBackend(compute_unit=ComputeUnit.CPU)
        handle = backend.load_model("/tmp/model.tflite")

        assert handle is not None
        mock_interpreter_class.assert_called_once()
        mock_interp.allocate_tensors.assert_called_once()

    @patch("moment_to_action.hardware._platforms._runtimes._litert._Interpreter")
    def test_litert_load_model_caching(self, mock_interpreter_class: Mock) -> None:
        """Test LiteRTBackend caches loaded models by path."""
        mock_interp = MagicMock()
        mock_interpreter_class.return_value = mock_interp

        backend = LiteRTBackend(compute_unit=ComputeUnit.CPU)
        handle1 = backend.load_model("/tmp/model.tflite")
        handle2 = backend.load_model("/tmp/model.tflite")

        assert handle1 is handle2
        # allocate_tensors called only once due to caching
        assert mock_interp.allocate_tensors.call_count == 1

    @patch("moment_to_action.hardware._platforms._runtimes._litert._Interpreter")
    def test_litert_run_mocked(self, mock_interpreter_class: Mock) -> None:
        """Test LiteRTBackend.run with mocked interpreter."""
        mock_interp = MagicMock()
        mock_interpreter_class.return_value = mock_interp

        # Setup mock output details and tensor
        mock_interp.get_output_details.return_value = [{"index": 0}]
        output_tensor = np.array([1.0, 2.0, 3.0])
        mock_interp.get_tensor.return_value = output_tensor

        backend = LiteRTBackend(compute_unit=ComputeUnit.CPU)
        handle = backend.load_model("/tmp/model.tflite")

        # Create input tensor
        input_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
        outputs = backend.run(handle, input_tensor)

        assert len(outputs) == 1
        np.testing.assert_array_equal(outputs[0], output_tensor)
        mock_interp.invoke.assert_called_once()

    @patch("moment_to_action.hardware._platforms._runtimes._litert._Interpreter")
    def test_litert_get_input_details_mocked(self, mock_interpreter_class: Mock) -> None:
        """Test LiteRTBackend.get_input_details."""
        mock_interp = MagicMock()
        mock_interpreter_class.return_value = mock_interp

        input_details = [
            {"name": "input", "shape": (1, 224, 224, 3), "dtype": np.float32, "index": 0}
        ]
        mock_interp.get_input_details.return_value = input_details

        backend = LiteRTBackend(compute_unit=ComputeUnit.CPU)
        handle = backend.load_model("/tmp/model.tflite")
        details = backend.get_input_details(handle)

        assert details == input_details

    @patch("moment_to_action.hardware._platforms._runtimes._litert._Interpreter")
    def test_litert_get_output_details_mocked(self, mock_interpreter_class: Mock) -> None:
        """Test LiteRTBackend.get_output_details."""
        mock_interp = MagicMock()
        mock_interpreter_class.return_value = mock_interp

        output_details = [{"name": "output", "shape": (1, 1000), "dtype": np.float32, "index": 0}]
        mock_interp.get_output_details.return_value = output_details

        backend = LiteRTBackend(compute_unit=ComputeUnit.CPU)
        handle = backend.load_model("/tmp/model.tflite")
        details = backend.get_output_details(handle)

        assert details == output_details

    def test_litert_import_error_fallback(self) -> None:
        """Test _have_ai_edge_litert=False fallback when ImportError occurs."""
        with patch.dict("sys.modules", {"ai_edge_litert": None}):
            import moment_to_action.hardware._platforms._runtimes._litert as litert_module

            importlib.reload(litert_module)
            # Verify module fell back to tf.lite
            assert hasattr(litert_module, "_have_ai_edge_litert")

    @patch("moment_to_action.hardware._platforms._runtimes._litert._Interpreter")
    def test_litert_load_interpreter_delegate_failure(self, mock_interpreter_class: Mock) -> None:
        """Test _load_interpreter raises RuntimeError when delegate fails."""
        mock_interpreter_class.side_effect = RuntimeError("Delegate failed to apply")

        backend = LiteRTBackend(compute_unit=ComputeUnit.NPU)
        # Mock _get_delegates to return a non-empty list via patch
        with patch.object(backend, "_get_delegates", return_value=["mock_delegate"]):
            with pytest.raises(RuntimeError, match="Delegate failed"):
                backend._load_interpreter("/tmp/model.tflite")

    @patch("moment_to_action.hardware._platforms._runtimes._litert._Interpreter")
    def test_litert_set_inputs_key_error_multi_input(self, mock_interpreter_class: Mock) -> None:
        """Test _set_inputs raises KeyError for missing input name in multi-input model."""
        mock_interp = MagicMock()
        mock_interpreter_class.return_value = mock_interp

        # Setup a multi-input model with two inputs
        input_details = [
            {"name": "input1", "dtype": np.float32, "index": 0},
            {"name": "input2", "dtype": np.float32, "index": 1},
        ]
        mock_interp.get_input_details.return_value = input_details

        backend = LiteRTBackend(compute_unit=ComputeUnit.CPU)
        handle = backend.load_model("/tmp/model.tflite")

        # Try to set input with wrong name
        inputs = {
            "wrong_name": np.array([1.0, 2.0], dtype=np.float32),
        }

        with pytest.raises(KeyError, match="Input name 'wrong_name' not found"):
            backend.run(handle, inputs)

    @patch("moment_to_action.hardware._platforms._runtimes._litert._Interpreter")
    def test_litert_set_inputs_type_error_dtype_mismatch(
        self, mock_interpreter_class: Mock
    ) -> None:
        """Test _set_inputs raises TypeError for dtype mismatch in multi-input model."""
        mock_interp = MagicMock()
        mock_interpreter_class.return_value = mock_interp

        # Setup a multi-input model expecting float32
        input_details = [
            {"name": "input1", "dtype": np.float32, "index": 0},
            {"name": "input2", "dtype": np.float32, "index": 1},
        ]
        mock_interp.get_input_details.return_value = input_details

        backend = LiteRTBackend(compute_unit=ComputeUnit.CPU)
        handle = backend.load_model("/tmp/model.tflite")

        # Try to set input with wrong dtype (int32 instead of float32)
        inputs = {
            "input1": np.array([1, 2], dtype=np.int32),
        }

        with pytest.raises(TypeError, match="Input 'input1' dtype mismatch"):
            backend.run(handle, inputs)

    @patch("moment_to_action.hardware._platforms._runtimes._litert._Interpreter")
    def test_litert_run_multi_input_happy_path(self, mock_interpreter_class: Mock) -> None:
        """Test multi-input run succeeds and sets all tensors correctly."""
        mock_interp = MagicMock()
        mock_interpreter_class.return_value = mock_interp

        input_details = [
            {"name": "input1", "dtype": np.float32, "index": 0},
            {"name": "input2", "dtype": np.float32, "index": 1},
        ]
        mock_interp.get_input_details.return_value = input_details
        mock_interp.get_output_details.return_value = [{"index": 0}]
        output_tensor = np.array([1.0])
        mock_interp.get_tensor.return_value = output_tensor

        backend = LiteRTBackend(compute_unit=ComputeUnit.CPU)
        handle = backend.load_model("/tmp/model.tflite")

        inputs = {
            "input1": np.array([1.0, 2.0], dtype=np.float32),
            "input2": np.array([3.0, 4.0], dtype=np.float32),
        }
        outputs = backend.run(handle, inputs)

        assert len(outputs) == 1
        assert mock_interp.set_tensor.call_count == 2


@pytest.mark.unit
class TestONNXBackend:
    """Test ONNX Runtime backend construction and methods."""

    def test_onnx_backend_construction(self) -> None:
        """Test ONNXBackend construction."""
        with patch("moment_to_action.hardware._platforms._runtimes._onnx.ort"):
            backend = ONNXBackend()
            assert backend is not None

    def test_onnx_get_supported_unit_returns_cpu(self) -> None:
        """Test ONNXBackend.get_supported_unit returns CPU."""
        backend = ONNXBackend()
        assert backend.get_supported_unit() == ComputeUnit.CPU

    @patch("moment_to_action.hardware._platforms._runtimes._onnx.ort.InferenceSession")
    def test_onnx_load_model_mocked(self, mock_session_class: Mock) -> None:
        """Test ONNXBackend.load_model with mocked session."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        backend = ONNXBackend()
        handle = backend.load_model("/tmp/model.onnx")

        assert handle is not None
        mock_session_class.assert_called_once()
        assert mock_session_class.call_args[0][0] == "/tmp/model.onnx"

    @patch("moment_to_action.hardware._platforms._runtimes._onnx.ort.InferenceSession")
    def test_onnx_load_model_caching(self, mock_session_class: Mock) -> None:
        """Test ONNXBackend caches loaded models by path."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        backend = ONNXBackend()
        handle1 = backend.load_model("/tmp/model.onnx")
        handle2 = backend.load_model("/tmp/model.onnx")

        assert handle1 is handle2
        # Session created only once due to caching
        assert mock_session_class.call_count == 1

    @patch("moment_to_action.hardware._platforms._runtimes._onnx.ort.InferenceSession")
    def test_onnx_run_mocked(self, mock_session_class: Mock) -> None:
        """Test ONNXBackend.run with mocked session."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        # Setup mock inputs and outputs
        mock_input = MagicMock()
        mock_input.name = "input_tensor"
        mock_session.get_inputs.return_value = [mock_input]

        output_tensor = np.array([1.0, 2.0, 3.0])
        mock_session.run.return_value = [output_tensor]

        backend = ONNXBackend()
        handle = backend.load_model("/tmp/model.onnx")

        # Create input tensor
        input_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
        outputs = backend.run(handle, input_tensor)

        assert len(outputs) == 1
        np.testing.assert_array_equal(outputs[0], output_tensor)
        mock_session.run.assert_called_once()

    @patch("moment_to_action.hardware._platforms._runtimes._onnx.ort.InferenceSession")
    def test_onnx_get_input_details_mocked(self, mock_session_class: Mock) -> None:
        """Test ONNXBackend.get_input_details."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_input = MagicMock()
        mock_input.name = "input"
        mock_input.shape = (1, 224, 224, 3)
        mock_input.type = "tensor(float)"
        mock_session.get_inputs.return_value = [mock_input]

        backend = ONNXBackend()
        handle = backend.load_model("/tmp/model.onnx")
        details = backend.get_input_details(handle)

        assert len(details) == 1
        assert details[0]["name"] == "input"
        assert details[0]["shape"] == (1, 224, 224, 3)

    @patch("moment_to_action.hardware._platforms._runtimes._onnx.ort.InferenceSession")
    def test_onnx_get_output_details_mocked(self, mock_session_class: Mock) -> None:
        """Test ONNXBackend.get_output_details."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_output = MagicMock()
        mock_output.name = "output"
        mock_output.shape = (1, 1000)
        mock_output.type = "tensor(float)"
        mock_session.get_outputs.return_value = [mock_output]

        backend = ONNXBackend()
        handle = backend.load_model("/tmp/model.onnx")
        details = backend.get_output_details(handle)

        assert len(details) == 1
        assert details[0]["name"] == "output"
        assert details[0]["shape"] == (1, 1000)
