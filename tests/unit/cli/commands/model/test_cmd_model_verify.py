"""Unit tests for m2a model verify command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner, Result

from moment_to_action.config import AppConfig
from moment_to_action.models import ModelID
from moment_to_action.models.image.detection._types import BoundingBox, Detection


def _make_ref_outputs(ref_dir: Path, n_images: int = 2) -> None:
    """Write synthetic reference outputs to ref_dir."""
    ref_dir.mkdir(parents=True, exist_ok=True)
    inputs = np.zeros((n_images, 3, 640, 640), dtype=np.float32)
    np.save(str(ref_dir / "inputs.npy"), inputs)
    boxes = np.zeros((n_images, 1, 10, 4), dtype=np.float32)
    np.save(str(ref_dir / "outputs_0.npy"), boxes)
    scores = np.zeros((n_images, 1, 10), dtype=np.float32)
    np.save(str(ref_dir / "outputs_1.npy"), scores)
    classes = np.zeros((n_images, 1, 10), dtype=np.uint8)
    np.save(str(ref_dir / "outputs_2.npy"), classes)


def _patched_pm(tmp_path: Path, variant_dir: Path) -> MagicMock:
    pm = MagicMock()
    pm.app_config_file = tmp_path / "cfg.json"
    pm.cache.models.get_variant_dir.return_value = variant_dir
    return pm


def _invoke(
    args: list[str],
    tmp_path: Path,
    mock_mgr: MagicMock,
    variant_dir: Path | None = None,
    mock_backend_cls: MagicMock | None = None,
) -> Result:
    from moment_to_action._cli import cli

    vdir = variant_dir or (tmp_path / "variants" / "default")
    pm = _patched_pm(tmp_path, vdir)
    be = mock_backend_cls or MagicMock()
    with patch("moment_to_action._cli.init_logging"):
        with patch("moment_to_action._cli.PathManager", return_value=pm):
            with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                with patch(
                    "moment_to_action._cli.commands.cmd_model.cmd_verify.ModelManager",
                    return_value=mock_mgr,
                ):
                    with patch(
                        "moment_to_action._cli.commands.cmd_model.cmd_verify.ComputeBackend",
                        return_value=be,
                    ):
                        return CliRunner().invoke(cli, ["model", "verify", *args])


def _make_model_mgr(detections: list[Detection] | None = None) -> MagicMock:
    if detections is None:
        detections = []
    mock_model = MagicMock()
    mock_model.run.return_value = [
        np.zeros((1, 10, 4), dtype=np.float32),
        np.zeros((1, 10), dtype=np.float32),
        np.zeros((1, 10), dtype=np.uint8),
    ]
    mock_model.post_proc.return_value = detections
    mgr = MagicMock()
    mgr.get_model.return_value = mock_model
    mgr.is_available.return_value = True
    return mgr


@pytest.mark.unit
class TestModelVerifyCommand:
    """Tests for m2a model verify."""

    def test_missing_reference_outputs_errors(self, tmp_path: Path) -> None:
        """Exits non-zero when reference_outputs dir is absent."""
        mgr = _make_model_mgr()
        variant_dir = tmp_path / "variants" / "default"
        variant_dir.mkdir(parents=True)
        # no reference_outputs/ inside
        result = _invoke(["yolo_v8", "--backend", "cpu"], tmp_path, mgr, variant_dir)
        assert result.exit_code != 0

    def test_passes_when_outputs_match(self, tmp_path: Path) -> None:
        """Exits 0 when model outputs match reference within tolerance."""
        variant_dir = tmp_path / "variants" / "default"
        ref_dir = variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        mgr = _make_model_mgr()
        result = _invoke(["yolo_v8", "--backend", "cpu"], tmp_path, mgr, variant_dir)
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_fails_when_raw_diff_exceeds_tol(self, tmp_path: Path) -> None:
        """Exits non-zero when raw element-wise diff exceeds --tol."""
        variant_dir = tmp_path / "variants" / "default"
        ref_dir = variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        mock_model = MagicMock()
        mock_model.post_proc.return_value = []
        # Return outputs that differ significantly from zeros
        mock_model.run.return_value = [
            np.ones((1, 10, 4), dtype=np.float32) * 999.0,
            np.zeros((1, 10), dtype=np.float32),
            np.zeros((1, 10), dtype=np.uint8),
        ]
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model
        mgr.is_available.return_value = True

        result = _invoke(
            ["yolo_v8", "--backend", "cpu", "--tol", "0.001"], tmp_path, mgr, variant_dir
        )
        assert result.exit_code != 0
        assert "FAIL" in result.output

    def test_npu_skips_raw_diff(self, tmp_path: Path) -> None:
        """NPU backend does not fail on raw diff — decoded comparison only."""
        variant_dir = tmp_path / "variants" / "default"
        ref_dir = variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        mock_model = MagicMock()
        mock_model.post_proc.return_value = []
        # NPU returns large raw diff — should still pass (decoded check passes)
        mock_model.run.return_value = [
            np.ones((1, 10, 4), dtype=np.float32) * 999.0,
            np.zeros((1, 10), dtype=np.float32),
            np.zeros((1, 10), dtype=np.uint8),
        ]
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model
        mgr.is_available.return_value = True

        result = _invoke(["yolo_v8", "--backend", "npu"], tmp_path, mgr, variant_dir)
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_npu_no_dlc_variant_fails(self, tmp_path: Path) -> None:
        """NPU backend fails gracefully when no DLC variant is registered."""
        from unittest.mock import patch as _patch

        variant_dir = tmp_path / "variants" / "default"
        ref_dir = variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        mgr = MagicMock()
        # Patch MODEL_REGISTRY to return only ONNX variants
        mock_info = MagicMock()
        mock_info.variants = {}
        with _patch(
            "moment_to_action._cli.commands.cmd_model.cmd_verify.MODEL_REGISTRY",
            {ModelID.YOLO_V8: mock_info},
        ):
            result = _invoke(["yolo_v8", "--backend", "npu"], tmp_path, mgr, variant_dir)
        assert result.exit_code != 0
        assert "FAIL" in result.output

    def test_decoded_mismatch_fails(self, tmp_path: Path) -> None:
        """Exits non-zero when decoded detections don't match reference."""
        variant_dir = tmp_path / "variants" / "default"
        ref_dir = variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        ref_det = Detection(label="cat", confidence=0.8, bbox=BoundingBox(0, 0, 10, 10))
        act_det = Detection(label="dog", confidence=0.8, bbox=BoundingBox(0, 0, 10, 10))

        call_count = 0

        def _post_proc_side_effect(_raw: object) -> list[Detection]:
            nonlocal call_count
            call_count += 1
            # Alternate: first call returns ref label, second returns different
            return [ref_det] if call_count % 2 == 1 else [act_det]

        mock_model = MagicMock()
        mock_model.run.return_value = [np.zeros((1, 10, 4)), np.zeros((1, 10)), np.zeros((1, 10))]
        mock_model.post_proc.side_effect = _post_proc_side_effect
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model
        mgr.is_available.return_value = True

        result = _invoke(["yolo_v8", "--backend", "cpu"], tmp_path, mgr, variant_dir)
        assert result.exit_code != 0
        assert "FAIL" in result.output

    def test_all_backends_run_when_no_backend_specified(self, tmp_path: Path) -> None:
        """All three backends are tested when --backend is omitted."""
        variant_dir = tmp_path / "variants" / "default"
        ref_dir = variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        mgr = _make_model_mgr()
        result = _invoke(["yolo_v8"], tmp_path, mgr, variant_dir)
        assert "CPU" in result.output
        assert "GPU" in result.output
        assert "NPU" in result.output

    def test_npu_dlc_variant_not_cached_fails(self, tmp_path: Path) -> None:
        """NPU backend fails when DLC variant exists but is not cached."""
        variant_dir = tmp_path / "variants" / "default"
        ref_dir = variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        mgr = MagicMock()
        mgr.get_model.return_value = MagicMock()
        # is_available returns False — variant exists but not cached
        mgr.is_available.return_value = False

        result = _invoke(["yolo_v8", "--backend", "npu"], tmp_path, mgr, variant_dir)
        assert result.exit_code != 0
        assert "FAIL" in result.output
        assert "not cached" in result.output.lower()
