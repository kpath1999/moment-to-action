"""Unit tests for X86_64GPUBackend (CUDA)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType


@pytest.mark.unit
class TestX86_64GPUBackend:  # noqa: N801
    """Tests for X86_64GPUBackend (requires CUDA mock)."""

    def _make_backend(self) -> object:
        """Return an X86_64GPUBackend with CUDA mocked as available."""
        import sys

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            from moment_to_action.hardware._platforms.x86_64._gpu_backend import X86_64GPUBackend

            return X86_64GPUBackend()

    def test_raises_when_cuda_unavailable(self) -> None:
        """X86_64GPUBackend raises RuntimeError when CUDA is not available."""
        import sys

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            from moment_to_action.hardware._platforms.x86_64._gpu_backend import X86_64GPUBackend

            with pytest.raises(RuntimeError, match="CUDA not available"):
                X86_64GPUBackend()

    def test_construction_with_cuda(self) -> None:
        """X86_64GPUBackend constructs when CUDA is available."""
        backend = self._make_backend()
        assert backend is not None

    def test_unit_property(self) -> None:
        """Unit returns GPU."""
        backend = self._make_backend()
        assert backend.unit == ComputeUnit.GPU  # type: ignore[attr-defined]

    def test_supported_dtypes(self) -> None:
        """supported_dtypes includes FP16 and FP32."""
        backend = self._make_backend()
        fmts = backend.supported_dtypes  # type: ignore[attr-defined]
        assert DataType.FP16 in fmts
        assert DataType.FP32 in fmts

    def test_supported_formats_includes_torch_and_llama_cpp(self) -> None:
        """supported_formats includes TORCH and LLAMA_CPP."""
        backend = self._make_backend()
        fmts = backend.supported_formats  # type: ignore[attr-defined]
        assert ModelType.TORCH in fmts
        assert ModelType.LLAMA_CPP in fmts

    def test_supported_formats_excludes_tflite(self) -> None:
        """supported_formats does NOT include TFLITE."""
        backend = self._make_backend()
        assert ModelType.TFLITE not in backend.supported_formats  # type: ignore[attr-defined]

    def test_load_tflite_raises(self) -> None:
        """load_tflite raises NotImplementedError since TFLITE is not supported."""
        backend = self._make_backend()
        with pytest.raises(NotImplementedError):
            backend.load_tflite("/tmp/model.tflite")  # type: ignore[attr-defined]

    def test_load_torch_returns_model(self) -> None:
        """load_torch returns a TorchModel with GPU unit."""
        from moment_to_action.hardware._loaded_models import TorchModel

        backend = self._make_backend()
        mock_module = MagicMock()
        with patch("torch.load", return_value=mock_module):
            model = backend.load_torch("/tmp/model.pt")  # type: ignore[attr-defined]

        assert isinstance(model, TorchModel)
        assert model.unit == ComputeUnit.GPU

    def test_load_llama_cpp_returns_model(self) -> None:
        """load_llama_cpp calls _start_llama_model with cpu_only=False."""
        mock_model = MagicMock()
        backend = self._make_backend()
        with patch(
            "moment_to_action.hardware._loaded_models._llama._start_llama_model",
            return_value=mock_model,
        ) as mock_start:
            result = backend.load_llama_cpp(  # type: ignore[attr-defined]
                "/tmp/model.gguf", server_path="/usr/bin/llama-server", port=8080
            )

        mock_start.assert_called_once_with(
            path="/tmp/model.gguf",
            mmproj=None,
            server_path="/usr/bin/llama-server",
            port=8080,
            unit=ComputeUnit.GPU,
            cpu_only=False,
        )
        assert result is mock_model
