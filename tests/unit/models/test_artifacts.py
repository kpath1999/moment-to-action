"""Unit tests for resolve_backend_artifact."""

from __future__ import annotations

from pathlib import Path

import pytest

from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models._artifacts import resolve_backend_artifact


@pytest.mark.unit
class TestResolveBackendArtifact:
    """Tests for resolve_backend_artifact."""

    def test_npu_bin_preferred_over_dlc(self, tmp_path: Path) -> None:
        """model.npu.bin is returned when present for NPU unit."""
        bin_file = tmp_path / "model.npu.bin"
        bin_file.write_text("bin")
        (tmp_path / "model.dlc").write_text("dlc")
        result = resolve_backend_artifact(tmp_path, ComputeUnit.NPU)
        assert result == bin_file

    def test_dsp_uses_npu_bin(self, tmp_path: Path) -> None:
        """DSP unit uses model.npu.bin (shared HTP/DSP binary)."""
        bin_file = tmp_path / "model.npu.bin"
        bin_file.write_text("bin")
        (tmp_path / "model.dlc").write_text("dlc")
        result = resolve_backend_artifact(tmp_path, ComputeUnit.DSP)
        assert result == bin_file

    def test_npu_falls_back_to_dlc_when_bin_absent(self, tmp_path: Path) -> None:
        """NPU falls back to model.dlc when model.npu.bin is absent."""
        dlc = tmp_path / "model.dlc"
        dlc.write_text("dlc")
        result = resolve_backend_artifact(tmp_path, ComputeUnit.NPU)
        assert result == dlc

    def test_cpu_uses_dlc(self, tmp_path: Path) -> None:
        """CPU always uses model.dlc — context binaries are HTP-only."""
        dlc = tmp_path / "model.dlc"
        dlc.write_text("dlc")
        (tmp_path / "model.cpu.bin").write_text("bin")
        result = resolve_backend_artifact(tmp_path, ComputeUnit.CPU)
        assert result == dlc

    def test_gpu_uses_dlc(self, tmp_path: Path) -> None:
        """GPU always uses model.dlc — context binaries are HTP-only."""
        dlc = tmp_path / "model.dlc"
        dlc.write_text("dlc")
        (tmp_path / "model.gpu.bin").write_text("bin")
        result = resolve_backend_artifact(tmp_path, ComputeUnit.GPU)
        assert result == dlc

    def test_raises_when_neither_present(self, tmp_path: Path) -> None:
        """FileNotFoundError when no usable artifact exists."""
        with pytest.raises(FileNotFoundError, match="No artifact found"):
            resolve_backend_artifact(tmp_path, ComputeUnit.CPU)
