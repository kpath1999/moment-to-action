"""Abstract base classes for platform-agnostic inference backends and resource monitors.

All platform-specific implementations live under _platforms/<chip>/ and must
subclass these ABCs.  Code outside this package should depend on these interfaces,
not on concrete implementations, to stay portable across hardware platforms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import psutil

if TYPE_CHECKING:
    import os
    from pathlib import Path

    from moment_to_action.hardware._types import (
        ComputeUnit,
        ComputeUnitUsageSample,
        TorchExecutionPolicy,
    )

# Type alias: single tensor (most models) or named dict (multi-input models).
ModelInput = np.ndarray | dict[str, np.ndarray]


class ResourceMonitor(ABC):
    """Abstract resource monitor.  Reads power draw and utilisation for a given compute unit."""

    @abstractmethod
    def sample(self, unit: ComputeUnit) -> ComputeUnitUsageSample:
        """Return a resource measurement for *unit*.

        Args:
            unit: The compute unit to sample.

        Returns:
            A ``ComputeUnitUsageSample`` with current power and utilisation figures.
        """
        ...

    @staticmethod
    def used_memory_mb() -> float:
        """Return used system memory in megabytes.

        Uses ``total - available`` per psutil docs — more accurate than ``.used``
        because ``.available`` accounts for reclaimable cache pages.
        """
        vm = psutil.virtual_memory()
        return (vm.total - vm.available) / (1024 * 1024)


class InferenceBackend(ABC):
    """Abstract inference runtime.

    One concrete subclass per runtime (LiteRT, ONNX, …).  Each backend is
    responsible for exactly one ``ComputeUnit``; fallback logic lives in
    ``ComputeBackend``, not here.
    """

    @abstractmethod
    def load_model(self, path: str | os.PathLike[str]) -> object:
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

    def resolve_torch_policy(self, requested: str = "auto") -> TorchExecutionPolicy:
        """Resolve torch device/dtype policy for this backend.

        Platform backends that support torch should override this. Runtime-only
        backends (for example LiteRT/ONNX wrappers) can inherit this default.

        Args:
            requested: ``"auto"`` or a string accepted by ``torch.device``.

        Returns:
            A resolved torch execution policy.
        """
        msg = f"{type(self).__name__} does not implement torch policy resolution"
        raise NotImplementedError(msg)

    def load_model_dlc(self, path: Path) -> object:
        """Load a DLC model and initialize the HTP backend via QAIRT.

        Override in platform backends that support DLC (e.g. QCS6490).

        Args:
            path: Path to the ``.dlc`` model file.

        Returns:
            An opaque DLC model handle.

        Raises:
            NotImplementedError: On platforms that do not support DLC.
        """
        msg = f"{type(self).__name__} does not support DLC models"
        raise NotImplementedError(msg)

    def infer_dlc(self, handle: object, inputs: ModelInput) -> dict[str, np.ndarray]:
        """Run inference on a loaded DLC model.

        Override in platform backends that support DLC (e.g. QCS6490).

        Args:
            handle: Handle returned by :meth:`load_model_dlc`.
            inputs: Single input tensor, or a name→tensor dict for multi-input
                graphs (e.g. the Detectron2 ROI head's ``features`` +
                ``proposals_boxes``).

        Returns:
            Mapping of output tensor names to arrays.

        Raises:
            NotImplementedError: On platforms that do not support DLC.
        """
        msg = f"{type(self).__name__} does not support DLC inference"
        raise NotImplementedError(msg)

    def unload_dlc(self, handle: object) -> None:
        """Release DLC backend resources.

        Override in platform backends that support DLC (e.g. QCS6490).

        Args:
            handle: Handle returned by :meth:`load_model_dlc`.

        Raises:
            NotImplementedError: On platforms that do not support DLC.
        """
        msg = f"{type(self).__name__} does not support DLC unloading"
        raise NotImplementedError(msg)

    def unload_model(self, handle: object) -> None:  # noqa: B027
        """Release resources for a model loaded via :meth:`load_model`.

        The default implementation is a no-op: ONNX and LiteRT backends
        rely on garbage collection.  Override if explicit cleanup is needed.

        Args:
            handle: Handle returned by :meth:`load_model`.
        """
