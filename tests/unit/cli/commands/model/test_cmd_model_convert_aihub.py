"""Unit tests for m2a model convert-aihub command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner, Result

from moment_to_action.config import AppConfig


def _patched_pm(tmp_path: Path) -> MagicMock:
    pm = MagicMock()
    pm.app_config_file = tmp_path / "cfg.json"
    return pm


def _invoke(args: list[str], tmp_path: Path, env: dict[str, str] | None = None) -> Result:
    from moment_to_action._cli import cli

    runner = CliRunner()
    with patch("moment_to_action._cli.init_logging"):
        with patch("moment_to_action._cli.PathManager", return_value=_patched_pm(tmp_path)):
            with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                return runner.invoke(cli, ["model", "convert-aihub", *args], env=env or {})


@pytest.mark.unit
class TestConvertAihubCommand:
    """Tests for m2a model convert-aihub."""

    def test_missing_token_raises(self, tmp_path: Path) -> None:
        """Exits non-zero with helpful message when QAI_HUB_API_TOKEN is unset."""
        result = _invoke(
            ["yolo_v8", "-o", str(tmp_path / "out")],
            tmp_path,
            env={"QAI_HUB_API_TOKEN": "", "QAI_HUB_API_KEY": ""},
        )
        assert result.exit_code != 0
        assert "QAI_HUB_API_TOKEN" in result.output

    def test_unsupported_model_raises(self, tmp_path: Path) -> None:
        """Exits non-zero when the model is not in the AI Hub model map."""
        result = _invoke(
            ["mobilenet_v2", "-o", str(tmp_path / "out")],
            tmp_path,
            env={"QAI_HUB_API_TOKEN": "tok"},
        )
        assert result.exit_code != 0
        assert "not supported" in result.output.lower()

    def test_missing_qai_hub_models_raises(self, tmp_path: Path) -> None:
        """Exits non-zero with helpful message when qai-hub-models is not installed."""
        out = tmp_path / "out"
        with patch(
            "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._run_aihub_export",
            side_effect=Exception("qai_hub_models not installed"),
        ):
            result = _invoke(
                ["yolo_v8", "-o", str(out)],
                tmp_path,
                env={"QAI_HUB_API_TOKEN": "tok"},
            )
        assert result.exit_code != 0

    def test_successful_export_copies_dlc(self, tmp_path: Path) -> None:
        """On success, model.dlc is written to output_dir."""
        out = tmp_path / "out"
        out.mkdir()
        build_dir = tmp_path / "out" / "_aihub_build"
        subdir = build_dir / "yolov8_det-qnn_dlc-w8a8"
        subdir.mkdir(parents=True)
        dlc_src = subdir / "yolov8_det.dlc"
        dlc_src.write_text("dlc_data")

        def fake_export(**_kwargs: object) -> Path:
            return dlc_src

        with patch(
            "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._run_aihub_export",
            return_value=dlc_src,
        ):
            result = _invoke(
                ["yolo_v8", "-o", str(out)],
                tmp_path,
                env={"QAI_HUB_API_TOKEN": "tok"},
            )

        assert result.exit_code == 0, result.output
        assert (out / "model.dlc").exists()
        assert (out / "model.dlc").read_text() == "dlc_data"

    def test_successful_export_copies_sidecars(self, tmp_path: Path) -> None:
        """metadata.json and labels.txt are copied when present."""
        out = tmp_path / "out"
        out.mkdir()
        build_dir = tmp_path / "out" / "_aihub_build" / "sub"
        build_dir.mkdir(parents=True)
        dlc_src = build_dir / "yolov8_det.dlc"
        dlc_src.write_text("dlc")
        (build_dir / "metadata.json").write_text("{}")
        (build_dir / "labels.txt").write_text("person\n")

        with patch(
            "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._run_aihub_export",
            return_value=dlc_src,
        ):
            result = _invoke(
                ["yolo_v8", "-o", str(out)],
                tmp_path,
                env={"QAI_HUB_API_TOKEN": "tok"},
            )

        assert result.exit_code == 0, result.output
        assert (out / "metadata.json").read_text() == "{}"
        assert (out / "labels.txt").read_text() == "person\n"

    def test_no_dlc_found_raises(self, tmp_path: Path) -> None:
        """Exits non-zero when export produces no .dlc file (empty output dir)."""
        out = tmp_path / "out"
        out.mkdir()

        mock_qai_hub = MagicMock()
        mock_export_mod = MagicMock()
        mock_export_mod.export_model.return_value = None  # export produces nothing

        import importlib

        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _run_aihub_export,
        )

        with (
            patch.dict("sys.modules", {"qai_hub": mock_qai_hub}),
            patch.object(importlib, "import_module", return_value=mock_export_mod),
        ):
            with pytest.raises(Exception, match=r"No \.dlc file found"):
                _run_aihub_export(
                    model_id="yolov8_det",
                    precision="w8a8",
                    runtime="qnn_dlc",
                    chipset="qualcomm-qcs6490",
                    output_dir=out / "_build",
                    token="tok",
                )

    def test_check_token_raises_when_unset(self) -> None:
        """_check_token raises ClickException when token env vars are absent."""
        import click

        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _check_token,
        )

        with patch.dict(
            "os.environ", {"QAI_HUB_API_TOKEN": "", "QAI_HUB_API_KEY": ""}, clear=False
        ):
            with pytest.raises(click.ClickException, match="QAI_HUB_API_TOKEN"):
                _check_token()

    def test_check_token_returns_token(self) -> None:
        """_check_token returns token when QAI_HUB_API_TOKEN is set."""
        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _check_token,
        )

        with patch.dict("os.environ", {"QAI_HUB_API_TOKEN": "mytoken"}, clear=False):
            assert _check_token() == "mytoken"

    def test_run_aihub_export_qai_hub_not_installed(self, tmp_path: Path) -> None:
        """_run_aihub_export raises ClickException when qai_hub is not installed."""
        import click

        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _run_aihub_export,
        )

        with patch.dict("sys.modules", {"qai_hub": None}):
            with pytest.raises(click.ClickException, match="qai-hub-models"):
                _run_aihub_export(
                    model_id="yolov8_det",
                    precision="w8a8",
                    runtime="qnn_dlc",
                    chipset="qualcomm-qcs6490",
                    output_dir=tmp_path / "_build",
                    token="tok",
                )

    def test_run_aihub_export_model_module_not_installed(self, tmp_path: Path) -> None:
        """_run_aihub_export raises ClickException when the model's extra is missing."""
        import importlib

        import click

        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _run_aihub_export,
        )

        mock_qai_hub = MagicMock()
        with (
            patch.dict("sys.modules", {"qai_hub": mock_qai_hub}),
            patch.object(importlib, "import_module", side_effect=ImportError("no extra")),
        ):
            with pytest.raises(click.ClickException, match="not available"):
                _run_aihub_export(
                    model_id="yolov8_det",
                    precision="w8a8",
                    runtime="qnn_dlc",
                    chipset="qualcomm-qcs6490",
                    output_dir=tmp_path / "_build",
                    token="tok",
                )

    def test_run_aihub_export_returns_dlc_path(self, tmp_path: Path) -> None:
        """_run_aihub_export returns the path to the produced .dlc file."""
        import importlib

        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _run_aihub_export,
        )

        build_dir = tmp_path / "_build"
        build_dir.mkdir(parents=True)
        dlc_file = build_dir / "yolov8_det.dlc"
        dlc_file.write_text("dlc_data")

        mock_qai_hub = MagicMock()
        mock_export_mod = MagicMock()
        mock_export_mod.export_model.return_value = None  # side-effect: dlc_file already exists

        with (
            patch.dict("sys.modules", {"qai_hub": mock_qai_hub}),
            patch.object(importlib, "import_module", return_value=mock_export_mod),
        ):
            result = _run_aihub_export(
                model_id="yolov8_det",
                precision="w8a8",
                runtime="qnn_dlc",
                chipset="qualcomm-qcs6490",
                output_dir=build_dir,
                token="tok",
            )
        assert result == dlc_file
