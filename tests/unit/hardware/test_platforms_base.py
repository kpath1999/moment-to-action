"""Unit tests for hardware._platforms._base abstract base classes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from moment_to_action.hardware._platforms._base import InferenceBackend, ResourceMonitor
from moment_to_action.hardware._types import ComputeUnit


@pytest.mark.unit
class TestResourceMonitorUsedMemory:
    """Tests for ResourceMonitor.used_memory_mb static method."""

    def test_used_memory_mb_returns_float(self) -> None:
        """used_memory_mb returns a non-negative float based on psutil."""
        mock_vm = MagicMock()
        mock_vm.total = 8 * 1024 * 1024 * 1024  # 8 GiB
        mock_vm.available = 2 * 1024 * 1024 * 1024  # 2 GiB available
        with patch("psutil.virtual_memory", return_value=mock_vm):
            result = ResourceMonitor.used_memory_mb()
        # 6 GiB used = 6144 MiB
        assert result == pytest.approx(6144.0)


@pytest.mark.unit
class TestInferenceBackendDefaults:
    """Tests for InferenceBackend default non-abstract method implementations."""

    def _make_backend(self) -> InferenceBackend:
        """Return a minimal concrete InferenceBackend."""

        class _Concrete(InferenceBackend):
            def load_model(self, path: object) -> object:
                """Load model."""
                return None

            def run(self, handle: object, inputs: object) -> list[np.ndarray]:
                """Run model."""
                return []

            def get_input_details(self, handle: object) -> list[dict]:
                """Get input details."""
                return []

            def get_output_details(self, handle: object) -> list[dict]:
                """Get output details."""
                return []

            def get_supported_unit(self) -> ComputeUnit:
                """Get unit."""
                return ComputeUnit.CPU

        return _Concrete()

    def test_resolve_torch_policy_raises(self) -> None:
        """resolve_torch_policy raises NotImplementedError by default."""
        b = self._make_backend()
        with pytest.raises(NotImplementedError, match="torch policy"):
            b.resolve_torch_policy()

    def test_load_model_dlc_raises(self) -> None:
        """load_model_dlc raises NotImplementedError by default."""
        b = self._make_backend()
        with pytest.raises(NotImplementedError, match="DLC models"):
            b.load_model_dlc(Path("/tmp/m.dlc"))

    def test_infer_dlc_raises(self) -> None:
        """infer_dlc raises NotImplementedError by default."""
        b = self._make_backend()
        with pytest.raises(NotImplementedError, match="DLC inference"):
            b.infer_dlc(None, np.array([]))

    def test_unload_dlc_raises(self) -> None:
        """unload_dlc raises NotImplementedError by default."""
        b = self._make_backend()
        with pytest.raises(NotImplementedError, match="DLC unloading"):
            b.unload_dlc(None)

    def test_unload_model_noop(self) -> None:
        """unload_model default is a no-op (does not raise)."""
        b = self._make_backend()
        b.unload_model(None)  # should not raise
