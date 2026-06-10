"""Unit tests for resolve_backend_artifact."""

from __future__ import annotations

from pathlib import Path

import pytest

from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models._artifacts import _BIN_BY_UNIT, resolve_backend_artifact


@pytest.mark.unit
class TestResolveBackendArtifact:
    """Tests for resolve_backend_artifact."""

    def test_cpu_bin_preferred_over_dlc(self, tmp_path: Path) -> None:
        """model.cpu.bin is returned when present for CPU unit."""
        bin_file = tmp_path / "model.cpu.bin"
        bin_file.write_text("bin")
        (tmp_path / "model.dlc").write_text("dlc")
        result = resolve_backend_artifact(tmp_path, ComputeUnit.CPU)
        assert result == bin_file

    def test_gpu_bin_preferred_over_dlc(self, tmp_path: Path) -> None:
        """model.gpu.bin is returned when present for GPU unit."""
        bin_file = tmp_path / "model.gpu.bin"
        bin_file.write_text("bin")
        (tmp_path / "model.dlc").write_text("dlc")
        result = resolve_backend_artifact(tmp_path, ComputeUnit.GPU)
        assert result == bin_file

    def test_npu_bin_preferred_over_dlc(self, tmp_path: Path) -> None:
        """model.npu.bin is returned when present for NPU unit."""
        bin_file = tmp_path / "model.npu.bin"
        bin_file.write_text("bin")
        (tmp_path / "model.dlc").write_text("dlc")
        result = resolve_backend_artifact(tmp_path, ComputeUnit.NPU)
        assert result == bin_file

    def test_dsp_falls_back_to_npu_bin(self, tmp_path: Path) -> None:
        """DSP unit uses model.npu.bin (shared HTP/DSP binary)."""
        bin_file = tmp_path / "model.npu.bin"
        bin_file.write_text("bin")
        result = resolve_backend_artifact(tmp_path, ComputeUnit.DSP)
        assert result == bin_file

    def test_falls_back_to_dlc_when_bin_absent(self, tmp_path: Path) -> None:
        """Falls back to model.dlc when no context binary is present."""
        dlc = tmp_path / "model.dlc"
        dlc.write_text("dlc")
        result = resolve_backend_artifact(tmp_path, ComputeUnit.NPU)
        assert result == dlc

    def test_raises_when_neither_present(self, tmp_path: Path) -> None:
        """FileNotFoundError when neither .bin nor .dlc exists."""
        with pytest.raises(FileNotFoundError, match="No artifact found"):
            resolve_backend_artifact(tmp_path, ComputeUnit.CPU)

    def test_bin_map_covers_all_units(self) -> None:
        """Every ComputeUnit has an entry in _BIN_BY_UNIT."""
        for unit in ComputeUnit:
            assert unit in _BIN_BY_UNIT, f"{unit} missing from _BIN_BY_UNIT"
