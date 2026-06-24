"""ComputeBackend — abstract per-unit inference backend (internal).

One concrete subclass per compute unit per platform:
  hardware/_platforms/qcs6490/_htp_backend.py  — QCS6490 NPU (DLC + TFLite/QNN)
  hardware/_platforms/qcs6490/_gpu_backend.py  — QCS6490 GPU (TFLite/delegate)
  hardware/_platforms/qcs6490/_cpu_backend.py  — QCS6490 CPU (TFLite + ONNX)
  hardware/_platforms/x86_64/_cpu_backend.py   — x86_64 CPU (TFLite + ONNX + DLC)
  hardware/_platforms/macos_arm64/_cpu_backend.py — macOS arm64 CPU

Not exported from ``hardware/__init__.py`` — callers use :class:`Platform`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    import os

    from moment_to_action.hardware._loaded_model import LoadedModel
    from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType


class ComputeBackend(ABC):
    """Abstract per-unit inference backend.

    Each subclass targets exactly one :class:`~moment_to_action.hardware.ComputeUnit`.
    Only the formats listed in :attr:`supported_formats` are implemented;
    all others raise :class:`NotImplementedError` via the default method bodies.
    """

    @property
    @abstractmethod
    def unit(self) -> ComputeUnit:
        """The compute unit this backend targets.

        Returns:
            The ``ComputeUnit`` this backend runs on.
        """
        ...

    @property
    @abstractmethod
    def supported_dtypes(self) -> set[DataType]:
        """Data types this backend can handle.

        Returns:
            Set of supported ``DataType`` members.
        """
        ...

    @property
    @abstractmethod
    def supported_formats(self) -> set[ModelType]:
        """Model formats this backend can load.

        Returns:
            Set of supported ``ModelType`` members.
        """
        ...

    def load_onnx(self, _path: str | os.PathLike[str]) -> LoadedModel:
        """Load an ONNX model.

        Args:
            _path: Path to the ``.onnx`` file.

        Returns:
            A :class:`~moment_to_action.hardware.LoadedModel` for this model.

        Raises:
            NotImplementedError: If ONNX is not in :attr:`supported_formats`.
        """
        self._raise_unsupported("ONNX")

    def load_dlc(self, _path: str | os.PathLike[str]) -> LoadedModel:
        """Load a DLC model.

        Args:
            _path: Path to the ``.dlc`` file.

        Returns:
            A :class:`~moment_to_action.hardware.LoadedModel` for this model.

        Raises:
            NotImplementedError: If DLC is not in :attr:`supported_formats`.
        """
        self._raise_unsupported("DLC")

    def load_torch(self, _path: str | os.PathLike[str]) -> LoadedModel:
        """Load a PyTorch model.

        Args:
            _path: Path to the saved model file.

        Returns:
            A :class:`~moment_to_action.hardware.LoadedModel` for this model.

        Raises:
            NotImplementedError: If TORCH is not in :attr:`supported_formats`.
        """
        self._raise_unsupported("TORCH")

    def load_tflite(self, _path: str | os.PathLike[str]) -> LoadedModel:
        """Load a TFLite model.

        Args:
            _path: Path to the ``.tflite`` file.

        Returns:
            A :class:`~moment_to_action.hardware.LoadedModel` for this model.

        Raises:
            NotImplementedError: If TFLITE is not in :attr:`supported_formats`.
        """
        self._raise_unsupported("TFLITE")

    def load_llama_cpp(
        self,
        path: str | os.PathLike[str],  # noqa: ARG002
        *,
        mmproj: str | os.PathLike[str] | None = None,  # noqa: ARG002
        server_path: str | os.PathLike[str] | None = None,  # noqa: ARG002
        port: int | None = None,  # noqa: ARG002
    ) -> LoadedModel:
        """Load a llama.cpp GGUF model.

        Args:
            path: Path to the ``.gguf`` model file.
            mmproj: Optional path to the multimodal projector file.
            server_path: Path to the ``llama-server`` binary. If ``None``,
                resolved by the backend from AppConfig or PATH.
            port: Port for llama-server. If ``None``, a free port is assigned.

        Returns:
            A :class:`~moment_to_action.hardware.LoadedModel` for this model.

        Raises:
            NotImplementedError: If LLAMA_CPP is not in :attr:`supported_formats`.
        """
        self._raise_unsupported("LLAMA_CPP")

    def _raise_unsupported(self, fmt: str) -> NoReturn:
        """Raise NotImplementedError for an unsupported format.

        Args:
            fmt: Human-readable format name (e.g. ``"ONNX"``).

        Raises:
            NotImplementedError: Always.
        """
        supported = ", ".join(f.value for f in self.supported_formats)
        msg = (
            f"{type(self).__name__} ({self.unit.name}) does not support {fmt} models. "
            f"Supported formats: {supported or 'none'}"
        )
        raise NotImplementedError(msg)
