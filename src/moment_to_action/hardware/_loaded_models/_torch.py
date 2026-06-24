"""PyTorch LoadedModel — shared across all platforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from moment_to_action.hardware._loaded_model import LoadedModel
from moment_to_action.hardware._types import ComputeUnit, DataType, ModelType

if TYPE_CHECKING:
    from collections.abc import Callable


class TorchModel(LoadedModel):
    """A PyTorch model loaded on a compute unit.

    Attributes:
        _unit: The compute unit this model runs on.
        _model: The loaded ``torch.nn.Module`` handle.
        _unloaded: Whether :meth:`unload` has already been called.
    """

    def __init__(self, unit: ComputeUnit, model: Any) -> None:
        """Initialize a TorchModel.

        Args:
            unit: The compute unit this model runs on.
            model: A loaded ``torch.nn.Module`` (or any callable torch model).
        """
        self._unit = unit
        self._model: Any = model
        self._unloaded = False

    @property
    def unit(self) -> ComputeUnit:
        """Compute unit this model is resident on."""
        return self._unit

    @property
    def dtype(self) -> DataType:
        """Data type — FP32 by default."""
        return DataType.FP32

    @property
    def model_type(self) -> ModelType:
        """Model format — always TORCH."""
        return ModelType.TORCH

    def run(self, inputs: object) -> object:
        """Run PyTorch inference.

        Args:
            inputs: Input data passed directly to the model's ``__call__``.

        Returns:
            Model output.

        Raises:
            RuntimeError: If :meth:`unload` has been called.
        """
        if self._unloaded:
            msg = "TorchModel has been unloaded; cannot run inference"
            raise RuntimeError(msg)
        return cast("Callable[[object], object]", self._model)(inputs)

    def unload(self) -> None:
        """Release the model handle."""
        if not self._unloaded:
            self._model = None
            self._unloaded = True
