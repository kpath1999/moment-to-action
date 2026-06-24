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
from moment_to_action.models.image.detection._base import ImageDetectionModel


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
                        "moment_to_action._cli.commands.cmd_model.cmd_verify.Platform",
                        return_value=be,
                    ):
                        return CliRunner().invoke(cli, ["model", "verify", *args])


def _make_model_mgr(*, verify_result: tuple[bool, str] = (True, "")) -> MagicMock:
    """Build a mock manager whose model passes the ImageDetectionModel isinstance check."""
    mock_model = MagicMock(spec=ImageDetectionModel)
    mock_model.verify_outputs.return_value = verify_result
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
        result = _invoke(["yolo_v8", "default", "--backend", "cpu"], tmp_path, mgr, variant_dir)
        assert result.exit_code != 0

    def test_passes_when_outputs_match(self, tmp_path: Path) -> None:
        """Exits 0 when verify_outputs returns (True, '')."""
        variant_dir = tmp_path / "variants" / "default"
        ref_dir = variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        mgr = _make_model_mgr(verify_result=(True, ""))
        result = _invoke(["yolo_v8", "default", "--backend", "cpu"], tmp_path, mgr, variant_dir)
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_fails_when_verify_returns_fail(self, tmp_path: Path) -> None:
        """Exits non-zero when verify_outputs returns (False, reason)."""
        variant_dir = tmp_path / "variants" / "default"
        ref_dir = variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        mgr = _make_model_mgr(verify_result=(False, "max_err=999"))
        result = _invoke(["yolo_v8", "default", "--backend", "cpu"], tmp_path, mgr, variant_dir)
        assert result.exit_code != 0
        assert "FAIL" in result.output

    def test_npu_backend(self, tmp_path: Path) -> None:
        """NPU backend runs and passes when model verifies."""
        variant_dir = tmp_path / "variants" / "qcs6490"
        ref_dir = variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        mgr = _make_model_mgr(verify_result=(True, ""))
        result = _invoke(["yolo_v8", "qcs6490", "--backend", "npu"], tmp_path, mgr, variant_dir)
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_non_detection_model_fails_gracefully(self, tmp_path: Path) -> None:
        """Non-ImageDetectionModel exits non-zero with 'does not support verify'."""
        variant_dir = tmp_path / "variants" / "default"
        ref_dir = variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        # Plain MagicMock is not an ImageDetectionModel
        mock_model = MagicMock()
        mgr = MagicMock()
        mgr.get_model.return_value = mock_model
        mgr.is_available.return_value = True

        result = _invoke(["yolo_v8", "default", "--backend", "cpu"], tmp_path, mgr, variant_dir)
        assert result.exit_code != 0
        assert "FAIL" in result.output

    def test_all_backends_run_when_no_backend_specified(self, tmp_path: Path) -> None:
        """All three backends are tested when --backend is omitted."""
        variant_dir = tmp_path / "variants" / "default"
        ref_dir = variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        mgr = _make_model_mgr(verify_result=(True, ""))
        result = _invoke(["yolo_v8", "default"], tmp_path, mgr, variant_dir)
        assert "CPU" in result.output
        assert "GPU" in result.output
        assert "NPU" in result.output

    def test_variant_not_cached_fails(self, tmp_path: Path) -> None:
        """Fails when requested variant is not cached."""
        variant_dir = tmp_path / "variants" / "qcs6490"
        ref_dir = variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        mgr = MagicMock()
        mgr.get_model.return_value = MagicMock(spec=ImageDetectionModel)
        mgr.is_available.return_value = False

        result = _invoke(["yolo_v8", "qcs6490", "--backend", "npu"], tmp_path, mgr, variant_dir)
        assert result.exit_code != 0
        assert "FAIL" in result.output
        assert "not cached" in result.output.lower()

    def test_variant_loads_ref_from_variant_dir(self, tmp_path: Path) -> None:
        """Positional variant loads reference outputs from that variant's directory."""
        qcs_variant_dir = tmp_path / "variants" / "qcs6490"
        ref_dir = qcs_variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        def _variant_dir(_model_id: str, variant: str) -> Path:
            return tmp_path / "variants" / variant

        pm = MagicMock()
        pm.app_config_file = tmp_path / "cfg.json"
        pm.cache.models.get_variant_dir.side_effect = _variant_dir

        mgr = _make_model_mgr(verify_result=(True, ""))

        from unittest.mock import patch as _patch

        from moment_to_action._cli import cli

        with _patch("moment_to_action._cli.init_logging"):
            with _patch("moment_to_action._cli.PathManager", return_value=pm):
                with _patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    with _patch(
                        "moment_to_action._cli.commands.cmd_model.cmd_verify.ModelManager",
                        return_value=mgr,
                    ):
                        with _patch(
                            "moment_to_action._cli.commands.cmd_model.cmd_verify.Platform",
                            return_value=MagicMock(),
                        ):
                            result = CliRunner().invoke(
                                cli,
                                ["model", "verify", "yolo_v8", "qcs6490", "--backend", "cpu"],
                            )
        assert result.exit_code == 0, result.output
        assert "PASS" in result.output
        pm.cache.models.get_variant_dir.assert_any_call("yolo_v8", "qcs6490")

    def test_variant_used_for_all_backends(self, tmp_path: Path) -> None:
        """Positional variant is used for every requested backend."""
        qcs_variant_dir = tmp_path / "variants" / "qcs6490"
        ref_dir = qcs_variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        def _variant_dir(_model_id: str, variant: str) -> Path:
            return tmp_path / "variants" / variant

        pm = MagicMock()
        pm.app_config_file = tmp_path / "cfg.json"
        pm.cache.models.get_variant_dir.side_effect = _variant_dir

        mgr = _make_model_mgr(verify_result=(True, ""))

        from unittest.mock import patch as _patch

        from moment_to_action._cli import cli

        with _patch("moment_to_action._cli.init_logging"):
            with _patch("moment_to_action._cli.PathManager", return_value=pm):
                with _patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    with _patch(
                        "moment_to_action._cli.commands.cmd_model.cmd_verify.ModelManager",
                        return_value=mgr,
                    ):
                        with _patch(
                            "moment_to_action._cli.commands.cmd_model.cmd_verify.Platform",
                            return_value=MagicMock(),
                        ):
                            result = CliRunner().invoke(
                                cli, ["model", "verify", "yolo_v8", "qcs6490"]
                            )
        assert "CPU" in result.output
        assert "GPU" in result.output
        assert "NPU" in result.output
        for call in mgr.get_model.call_args_list:
            assert call.kwargs.get("variant") == "qcs6490"

    def test_variant_not_cached_fails_gracefully(self, tmp_path: Path) -> None:
        """Unavailable variant reports FAIL for every backend."""
        qcs_variant_dir = tmp_path / "variants" / "qcs6490"
        ref_dir = qcs_variant_dir / "reference_outputs"
        _make_ref_outputs(ref_dir)

        def _variant_dir(_model_id: str, variant: str) -> Path:
            return tmp_path / "variants" / variant

        pm = MagicMock()
        pm.app_config_file = tmp_path / "cfg.json"
        pm.cache.models.get_variant_dir.side_effect = _variant_dir

        mgr = MagicMock()
        mgr.is_available.return_value = False

        from unittest.mock import patch as _patch

        from moment_to_action._cli import cli

        with _patch("moment_to_action._cli.init_logging"):
            with _patch("moment_to_action._cli.PathManager", return_value=pm):
                with _patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    with _patch(
                        "moment_to_action._cli.commands.cmd_model.cmd_verify.ModelManager",
                        return_value=mgr,
                    ):
                        with _patch(
                            "moment_to_action._cli.commands.cmd_model.cmd_verify.Platform",
                            return_value=MagicMock(),
                        ):
                            result = CliRunner().invoke(
                                cli,
                                ["model", "verify", "yolo_v8", "qcs6490", "--backend", "cpu"],
                            )
        assert result.exit_code != 0
        assert "FAIL" in result.output
