"""Abstract base classes for platform-agnostic inference backends and power monitors.

All platform-specific implementations live under _platforms/<chip>/ and must
subclass these ABCs.  Code outside this package should depend on these interfaces,
not on concrete implementations, to stay portable across hardware platforms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from moment_to_action.hardware._types import ComputeUnit, PowerSample

# Type alias: single tensor (most models) or named dict (multi-input models).
ModelInput = np.ndarray | dict[str, np.ndarray]


class PowerMonitor(ABC):
    """Abstract power monitor.  Reads power draw for a given compute unit."""

    @abstractmethod
    def sample(self, unit: ComputeUnit) -> PowerSample:
        """Return a power measurement for *unit*.

        Args:
            unit: The compute unit to sample.

        Returns:
            A ``PowerSample`` with current power and utilisation figures.
        """
        ...


class InferenceBackend(ABC):
    """Abstract inference runtime.

    One concrete subclass per runtime (LiteRT, ONNX, …).  Each backend is
    responsible for exactly one ``ComputeUnit``; fallback logic lives in
    ``ComputeBackend``, not here.
    """

    @abstractmethod
    def load_model(self, path: str) -> object:
        """Load a model from *path* and return an opaque handle.

        Args:
            path: Filesystem path to the model file.

        Returns:
            A runtime-specific model handle (interpreter, session, …).
        """
        ...

    @abstractmethod
    def run(self, handle: object, inputs: ModelInput) -> list[np.ndarray]:
        """Run inference and return all output tensors.

        Args:
            handle: The handle returned by :meth:`load_model`.
            inputs: A single ndarray (single-input models) or a name→tensor
                dict (multi-input models).

        Returns:
            List of output tensors, one per model output slot.
        """
        ...

    @abstractmethod
    def get_input_details(self, handle: object) -> list[dict]:
        """Return metadata for each input tensor of the loaded model.

        Args:
            handle: The handle returned by :meth:`load_model`.

        Returns:
            List of dicts, one per input slot.  Dict keys are runtime-specific
            but at minimum include ``"name"``, ``"shape"``, and ``"dtype"``.
        """
        ...

    @abstractmethod
    def get_output_details(self, handle: object) -> list[dict]:
        """Return metadata for each output tensor of the loaded model.

        Args:
            handle: The handle returned by :meth:`load_model`.

        Returns:
            List of dicts, one per output slot.
        """
        ...

    @abstractmethod
    def get_supported_unit(self) -> ComputeUnit:
        """Return the ``ComputeUnit`` this backend targets."""
        ...
