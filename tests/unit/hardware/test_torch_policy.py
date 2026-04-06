"""Unit tests for resolve_torch_execution_policy."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from moment_to_action.hardware._platforms._runtimes._torch_policy import (
    resolve_torch_execution_policy,
)


@pytest.mark.unit
class TestResolveTorchExecutionPolicy:
    """Tests for resolve_torch_execution_policy()."""

    def test_explicit_cpu_device(self) -> None:
        """Explicit 'cpu' device returns cpu/float32 policy."""
        policy = resolve_torch_execution_policy("cpu")
        assert policy.device == torch.device("cpu")
        assert policy.dtype == torch.float32

    def test_auto_mode_returns_valid_policy(self) -> None:
        """Auto mode returns a valid policy (device and dtype are torch objects)."""
        policy = resolve_torch_execution_policy("auto")
        assert isinstance(policy.device, torch.device)
        assert policy.device.type in ("cpu", "cuda", "mps")
        assert policy.dtype in (torch.float32, torch.float16, torch.bfloat16)

    @patch("torch.cuda.is_available", new=lambda: True)
    def test_auto_selects_cuda_when_available(self) -> None:
        """Auto mode picks cuda when cuda is available."""
        policy = resolve_torch_execution_policy("auto")
        assert policy.device == torch.device("cuda")
        assert policy.dtype == torch.bfloat16

    @patch("torch.backends.mps.is_available", new=lambda: True)
    @patch("torch.cuda.is_available", new=lambda: False)
    def test_auto_selects_mps_when_available(self) -> None:
        """Auto mode picks mps when cuda unavailable but mps available."""
        policy = resolve_torch_execution_policy("auto")
        assert policy.device == torch.device("mps")
        assert policy.dtype == torch.float16

    @patch("torch.backends.mps.is_available", new=lambda: False)
    @patch("torch.cuda.is_available", new=lambda: False)
    def test_auto_falls_back_to_cpu(self) -> None:
        """Auto mode falls back to cpu when neither cuda nor mps available."""
        policy = resolve_torch_execution_policy("auto")
        assert policy.device == torch.device("cpu")
        assert policy.dtype == torch.float32

    @patch("torch.backends.mps.is_available", new=lambda: False)
    @patch("torch.cuda.is_available", new=lambda: False)
    def test_explicit_cpu_bypasses_auto_detection(self) -> None:
        """Explicit device string bypasses auto-detection entirely."""
        policy = resolve_torch_execution_policy("cpu")
        assert policy.device == torch.device("cpu")
        assert policy.dtype == torch.float32

    def test_policy_device_is_torch_device(self) -> None:
        """Policy device attribute is always a torch.device instance."""
        policy = resolve_torch_execution_policy("cpu")
        assert isinstance(policy.device, torch.device)

    def test_policy_dtype_is_torch_dtype(self) -> None:
        """Policy dtype attribute is always a torch.dtype instance."""
        policy = resolve_torch_execution_policy("cpu")
        assert isinstance(policy.dtype, torch.dtype)
