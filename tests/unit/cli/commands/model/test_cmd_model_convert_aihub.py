"""Unit tests for m2a model convert-aihub command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from click.testing import CliRunner, Result

from moment_to_action.config import AppConfig


def _patched_pm(tmp_path: Path) -> MagicMock:
    pm = MagicMock()
    pm.app_config_file = tmp_path / "cfg.json"
    return pm


def _calib_dir(tmp_path: Path) -> Path:
    """Create a minimal calibration dir with one fake JPEG."""
    d = tmp_path / "calib"
    d.mkdir(exist_ok=True)
    (d / "img.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    return d


def _invoke(
    args: list[str],
    tmp_path: Path,
    env: dict[str, str] | None = None,
    *,
    calib: bool = True,
) -> Result:
    from moment_to_action._cli import cli

    extra: list[str] = []
    if calib:
        extra = ["--calibration-dir", str(_calib_dir(tmp_path))]
    runner = CliRunner()
    with patch("moment_to_action._cli.init_logging"):
        with patch("moment_to_action._cli.PathManager", return_value=_patched_pm(tmp_path)):
            with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                return runner.invoke(cli, ["model", "convert-aihub", *args, *extra], env=env or {})


def _make_artifact(tmp_path: Path, ext: str) -> Path:
    """Create a fake artifact file and return its path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / f"artifact{ext}"
    p.write_text("data")
    return p


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
        """On success, model.dlc and model.npu.bin are written to output_dir."""
        out = tmp_path / "out"
        out.mkdir()

        dlc_src = _make_artifact(tmp_path / "dlc", ".dlc")
        npu_src = _make_artifact(tmp_path / "npu", ".bin")

        with (
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._run_aihub_export",
                side_effect=[dlc_src, npu_src],
            ),
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._capture_reference_outputs"
            ),
        ):
            result = _invoke(
                ["yolo_v8", "-o", str(out)],
                tmp_path,
                env={"QAI_HUB_API_TOKEN": "tok"},
            )

        assert result.exit_code == 0, result.output
        assert (out / "model.dlc").exists()
        assert (out / "model.npu.bin").exists()

    def test_successful_export_copies_sidecars(self, tmp_path: Path) -> None:
        """metadata.json and labels.txt are copied from the DLC build dir when present."""
        out = tmp_path / "out"
        out.mkdir()

        build_dlc_dir = tmp_path / "dlc"
        build_dlc_dir.mkdir()
        dlc_src = build_dlc_dir / "yolov8_det.dlc"
        dlc_src.write_text("dlc")
        (build_dlc_dir / "metadata.json").write_text("{}")
        (build_dlc_dir / "labels.txt").write_text("person\n")

        npu_src = _make_artifact(tmp_path / "npu", ".bin")

        with (
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._run_aihub_export",
                side_effect=[dlc_src, npu_src],
            ),
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._capture_reference_outputs"
            ),
        ):
            result = _invoke(
                ["yolo_v8", "-o", str(out)],
                tmp_path,
                env={"QAI_HUB_API_TOKEN": "tok"},
            )

        assert result.exit_code == 0, result.output
        assert (out / "metadata.json").read_text() == "{}"
        assert (out / "labels.txt").read_text() == "person\n"

    def test_run_aihub_export_calls_correct_runtimes(self, tmp_path: Path) -> None:
        """convert-aihub calls _run_aihub_export once for DLC then once for npu bin."""
        out = tmp_path / "out"
        out.mkdir()

        dlc_src = _make_artifact(tmp_path / "dlc", ".dlc")
        npu_src = _make_artifact(tmp_path / "npu", ".bin")

        with (
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._run_aihub_export",
                side_effect=[dlc_src, npu_src],
            ) as mock_export,
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._capture_reference_outputs"
            ) as mock_capture,
        ):
            result = _invoke(
                ["yolo_v8", "-o", str(out)],
                tmp_path,
                env={"QAI_HUB_API_TOKEN": "tok"},
            )

        assert result.exit_code == 0, result.output
        assert mock_export.call_count == 2
        # First call: DLC (always float — portable master artifact)
        assert mock_export.call_args_list[0] == call(
            model_id="yolov8_det",
            precision="float",
            runtime="qnn_dlc",
            chipset="qualcomm-qcs6490",
            output_dir=out / "_aihub_build" / "dlc",
            token="tok",
        )
        # Reference outputs captured after DLC but before context binary
        assert mock_capture.call_count == 1
        # Second call: context binary for NPU
        assert mock_export.call_args_list[1] == call(
            model_id="yolov8_det",
            precision="w8a8",
            runtime="qnn_context_binary",
            chipset="qualcomm-qcs6490",
            output_dir=out / "_aihub_build" / "npu",
            token="tok",
            compile_options="--qnn_options default_graph_htp_precision=FLOAT16",
            link_options="--qnn_options default_graph_htp_precision=FLOAT16",
        )

    def test_rf_detr_skips_npu_context_binary(self, tmp_path: Path) -> None:
        """RF-DETR skips context binary: float-only, float32 I/O unsupported on Hexagon v68."""
        out = tmp_path / "out"
        out.mkdir()

        dlc_src = _make_artifact(tmp_path / "dlc", ".dlc")

        with (
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._run_aihub_export",
                side_effect=[dlc_src],
            ) as mock_export,
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._capture_reference_outputs"
            ),
        ):
            result = _invoke(
                ["rf_detr", "-o", str(out)],
                tmp_path,
                env={"QAI_HUB_API_TOKEN": "tok"},
            )

        assert result.exit_code == 0, result.output
        # Only one export call (DLC); context binary skipped
        assert mock_export.call_count == 1
        assert not (out / "model.npu.bin").exists()
        assert "Skipping NPU context binary" in result.output

    def test_rtmdet_skips_npu_context_binary(self, tmp_path: Path) -> None:
        """RTMDet skips context binary: float decode head leaves boxes float, rejected on v68."""
        out = tmp_path / "out"
        out.mkdir()

        dlc_src = _make_artifact(tmp_path / "dlc", ".dlc")

        with (
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._run_aihub_export",
                side_effect=[dlc_src],
            ) as mock_export,
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._capture_reference_outputs"
            ),
        ):
            result = _invoke(
                ["rtm_det", "-o", str(out)],
                tmp_path,
                env={"QAI_HUB_API_TOKEN": "tok"},
            )

        assert result.exit_code == 0, result.output
        # Only one export call (DLC); context binary skipped
        assert mock_export.call_count == 1
        assert not (out / "model.npu.bin").exists()
        assert "Skipping NPU context binary" in result.output

    def test_detectron2_places_both_components(self, tmp_path: Path) -> None:
        """Detectron2 exports place per-component DLCs and context binaries by stem."""
        out = tmp_path / "out"
        out.mkdir()

        def _fake_export(**kwargs: object) -> Path:
            """Write per-component artifacts into the build dir and return one path."""
            build = Path(str(kwargs["output_dir"]))
            build.mkdir(parents=True, exist_ok=True)
            ext = ".bin" if kwargs["runtime"] == "qnn_context_binary" else ".dlc"
            for comp in ("proposal_generator", "roi_head"):
                (build / f"{comp}{ext}").write_text("data")
            return build / f"proposal_generator{ext}"

        with (
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._run_aihub_export",
                side_effect=_fake_export,
            ) as mock_export,
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._capture_reference_outputs"
            ),
        ):
            result = _invoke(
                ["detectron2", "--precision", "w8a16", "-o", str(out)],
                tmp_path,
                env={"QAI_HUB_API_TOKEN": "tok"},
            )

        assert result.exit_code == 0, result.output
        # Float component DLCs are produced transiently for reference capture, then
        # deleted (NPU-only ship): CPU/GPU use the single-graph ONNX default variant.
        assert not (out / "model.proposal_generator.dlc").exists()
        assert not (out / "model.roi_head.dlc").exists()
        assert (out / "model.proposal_generator.npu.bin").exists()
        assert (out / "model.roi_head.npu.bin").exists()
        # Single-graph names are NOT produced for a multi-component model
        assert not (out / "model.dlc").exists()
        assert not (out / "model.npu.bin").exists()
        # NPU step uses the CLI precision and no FLOAT16 options (full-int graph)
        assert mock_export.call_args_list[1] == call(
            model_id="detectron2_detection",
            precision="w8a16",
            runtime="qnn_context_binary",
            chipset="qualcomm-qcs6490",
            output_dir=out / "_aihub_build" / "npu",
            token="tok",
            compile_options="",
            link_options="",
        )

    def test_detectron2_w8a8_variant(self, tmp_path: Path) -> None:
        """Detectron2 honours --precision w8a8 for the context-binary step."""
        out = tmp_path / "out"
        out.mkdir()

        def _fake_export(**kwargs: object) -> Path:
            build = Path(str(kwargs["output_dir"]))
            build.mkdir(parents=True, exist_ok=True)
            ext = ".bin" if kwargs["runtime"] == "qnn_context_binary" else ".dlc"
            for comp in ("proposal_generator", "roi_head"):
                (build / f"{comp}{ext}").write_text("data")
            return build / f"roi_head{ext}"

        with (
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._run_aihub_export",
                side_effect=_fake_export,
            ) as mock_export,
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._capture_reference_outputs"
            ),
        ):
            result = _invoke(
                ["detectron2", "--precision", "w8a8", "-o", str(out)],
                tmp_path,
                env={"QAI_HUB_API_TOKEN": "tok"},
            )

        assert result.exit_code == 0, result.output
        assert mock_export.call_args_list[1].kwargs["precision"] == "w8a8"
        assert (out / "model.proposal_generator.npu.bin").exists()

    def test_detectron2_missing_component_artifact_raises(self, tmp_path: Path) -> None:
        """A missing component artifact in the build dir fails with a clear error."""
        out = tmp_path / "out"
        out.mkdir()

        def _fake_export(**kwargs: object) -> Path:
            build = Path(str(kwargs["output_dir"]))
            build.mkdir(parents=True, exist_ok=True)
            # Only write proposal_generator — roi_head is missing on purpose.
            p = build / "proposal_generator.dlc"
            p.write_text("data")
            return p

        with (
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._run_aihub_export",
                side_effect=_fake_export,
            ),
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._capture_reference_outputs"
            ),
        ):
            result = _invoke(
                ["detectron2", "--precision", "w8a16", "-o", str(out)],
                tmp_path,
                env={"QAI_HUB_API_TOKEN": "tok"},
            )

        assert result.exit_code != 0
        assert "roi_head" in result.output

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

    def test_no_bin_found_raises(self, tmp_path: Path) -> None:
        """Exits non-zero when context binary export produces no .bin file."""
        out = tmp_path / "out"
        out.mkdir()

        mock_qai_hub = MagicMock()
        mock_export_mod = MagicMock()
        mock_export_mod.export_model.return_value = None

        import importlib

        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _run_aihub_export,
        )

        with (
            patch.dict("sys.modules", {"qai_hub": mock_qai_hub}),
            patch.object(importlib, "import_module", return_value=mock_export_mod),
        ):
            with pytest.raises(Exception, match=r"No \.bin file found"):
                _run_aihub_export(
                    model_id="yolov8_det",
                    precision="w8a8",
                    runtime="qnn_context_binary",
                    chipset="qualcomm-qcs6490",
                    output_dir=out / "_build",
                    token="tok",
                )

    def test_link_options_monkey_patches_link_model(self, tmp_path: Path) -> None:
        """_run_aihub_export patches link_model with link_options and restores it."""
        import importlib
        import types

        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _run_aihub_export,
        )

        out = tmp_path / "_build"
        out.mkdir()
        bin_file = out / "model.bin"
        bin_file.write_bytes(b"bin")

        mock_qai_hub = MagicMock()
        original_link = MagicMock(return_value=None)
        captured: list[object] = []

        def _spy_link(
            compiled_model: object,
            device: object,
            model_name: str,
            model: object,
            target_runtime: object,
            extra_options: str = "",
        ) -> None:
            captured.append(extra_options)
            return original_link(
                compiled_model,
                device,
                model_name,
                model,
                target_runtime,
                extra_options=extra_options,
            )

        # export_model calls link_model internally — simulate that so the patch executes
        def _fake_export_model(**_kwargs: object) -> None:
            mock_export_mod.link_model(None, None, "m", None, None)

        mock_export_mod = types.SimpleNamespace(
            Precision=MagicMock(w8a8=MagicMock()),
            TargetRuntime=MagicMock(return_value=MagicMock()),
            export_model=_fake_export_model,
            link_model=_spy_link,
        )

        with (
            patch.dict("sys.modules", {"qai_hub": mock_qai_hub}),
            patch.object(importlib, "import_module", return_value=mock_export_mod),
        ):
            _run_aihub_export(
                model_id="yolov8_det",
                precision="w8a8",
                runtime="qnn_context_binary",
                chipset="qualcomm-qcs6490",
                output_dir=out,
                token="tok",
                compile_options="--qnn_options default_graph_htp_precision=FLOAT16",
                link_options="--qnn_options default_graph_htp_precision=FLOAT16",
            )

        # patched link_model injected link_options into extra_options
        assert captured == ["--qnn_options default_graph_htp_precision=FLOAT16"]
        # link_model is restored after the call
        assert mock_export_mod.link_model is _spy_link

    def test_check_token_raises_when_unset(self) -> None:
        """_check_token raises ClickException when token env vars are absent."""
        import click

        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _check_token,
        )

        env = {"QAI_HUB_API_TOKEN": "", "QAI_HUB_API_KEY": ""}
        with patch.dict("os.environ", env, clear=False):
            with pytest.raises(click.ClickException, match="QAI_HUB_API_TOKEN"):
                _check_token()

    def test_check_token_returns_token(self) -> None:
        """_check_token returns token when QAI_HUB_API_TOKEN is set."""
        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _check_token,
        )

        with patch.dict("os.environ", {"QAI_HUB_API_TOKEN": "mytoken"}, clear=False):
            assert _check_token() == "mytoken"

    def test_run_aihub_export_no_device_found(self, tmp_path: Path) -> None:
        """_run_aihub_export raises ClickException when no device matches the chipset."""
        import click

        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _run_aihub_export,
        )

        mock_qai_hub = MagicMock()
        mock_qai_hub.Client.return_value.get_devices.return_value = []
        with patch.dict("sys.modules", {"qai_hub": mock_qai_hub}):
            with pytest.raises(click.ClickException, match="No AI Hub device"):
                _run_aihub_export(
                    model_id="yolov8_det",
                    precision="w8a8",
                    runtime="qnn_dlc",
                    chipset="qualcomm-qcs9999",
                    output_dir=tmp_path / "_build",
                    token="tok",
                )

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

    def test_run_aihub_export_returns_bin_path(self, tmp_path: Path) -> None:
        """_run_aihub_export returns the .bin path for qnn_context_binary runtime."""
        import importlib

        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _run_aihub_export,
        )

        build_dir = tmp_path / "_build"
        build_dir.mkdir(parents=True)
        bin_file = build_dir / "yolov8_det.bin"
        bin_file.write_text("bin_data")

        mock_qai_hub = MagicMock()
        mock_export_mod = MagicMock()
        mock_export_mod.export_model.return_value = None

        with (
            patch.dict("sys.modules", {"qai_hub": mock_qai_hub}),
            patch.object(importlib, "import_module", return_value=mock_export_mod),
        ):
            result = _run_aihub_export(
                model_id="yolov8_det",
                precision="w8a8",
                runtime="qnn_context_binary",
                chipset="qualcomm-qcs6490",
                output_dir=build_dir,
                token="tok",
            )
        assert result == bin_file

    def test_capture_reference_outputs_no_images_raises(self, tmp_path: Path) -> None:
        """_capture_reference_outputs raises ClickException when calibration dir is empty."""
        import click

        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _capture_reference_outputs,
        )

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        from moment_to_action.models import ModelID

        with pytest.raises(click.ClickException, match="No images found"):
            _capture_reference_outputs(ModelID.YOLO_V8, empty_dir, tmp_path / "out")

    def test_capture_reference_outputs_saves_files(self, tmp_path: Path) -> None:
        """_capture_reference_outputs writes inputs.npy and outputs_0/1/2.npy from DLC."""
        import numpy as np

        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _capture_reference_outputs,
        )
        from moment_to_action.models import ModelID
        from moment_to_action.models.image._base import ImageModel

        calib = tmp_path / "calib"
        calib.mkdir()
        (calib / "img.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)

        mock_model = MagicMock(spec=ImageModel)
        mock_model.prepare.return_value = np.zeros((1, 640, 640, 3), dtype=np.float32)
        mock_model.run.return_value = [
            np.zeros((1, 8400, 4), dtype=np.float32),
            np.zeros((1, 8400), dtype=np.float32),
            np.zeros((1, 8400), dtype=np.int64),
        ]

        out = tmp_path / "out"

        with (
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._build_dlc_model",
                return_value=mock_model,
            ),
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub.cv2.imread",
                return_value=np.zeros((480, 640, 3), dtype=np.uint8),
            ),
        ):
            _capture_reference_outputs(ModelID.YOLO_V8, calib, out)

        ref_dir = out / "reference_outputs"
        assert (ref_dir / "inputs.npy").exists()
        assert (ref_dir / "outputs_0.npy").exists()
        assert (ref_dir / "outputs_1.npy").exists()
        assert (ref_dir / "outputs_2.npy").exists()

    def test_capture_reference_outputs_not_image_model_raises(self, tmp_path: Path) -> None:
        """_capture_reference_outputs raises ClickException for non-image models."""
        import click

        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _capture_reference_outputs,
        )
        from moment_to_action.models import ModelID

        calib = tmp_path / "calib"
        calib.mkdir()
        (calib / "img.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)

        with (
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub._build_dlc_model",
                return_value=MagicMock(spec=object),
            ),
            patch(
                "moment_to_action._cli.commands.cmd_model.cmd_convert_aihub.cv2.imread",
                return_value=__import__("numpy").zeros((480, 640, 3), dtype="uint8"),
            ),
        ):
            with pytest.raises(click.ClickException, match="not an image model"):
                _capture_reference_outputs(ModelID.YOLO_V8, calib, tmp_path / "out")

    def test_build_dlc_model_yolo_returns_yolo_model(self, tmp_path: Path) -> None:
        """_build_dlc_model returns a YOLOModel with qcs6490/NHWC config for YOLO_V8."""
        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import _build_dlc_model
        from moment_to_action.models import ModelID
        from moment_to_action.models.image.detection.yolo._model import YOLOModel

        model = _build_dlc_model(ModelID.YOLO_V8, tmp_path)
        assert isinstance(model, YOLOModel)
        assert model.input_layout == "NHWC"

    def test_build_dlc_model_rf_detr_returns_rfdetr_model(self, tmp_path: Path) -> None:
        """_build_dlc_model returns an RFDETRModel with qcs6490/NHWC config for RF_DETR."""
        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import _build_dlc_model
        from moment_to_action.models import ModelID
        from moment_to_action.models.image.detection.rf_detr._model import RFDETRModel

        model = _build_dlc_model(ModelID.RF_DETR, tmp_path)
        assert isinstance(model, RFDETRModel)
        assert model.input_layout == "NHWC"

    def test_build_dlc_model_rtm_det_returns_rtmdet_model(self, tmp_path: Path) -> None:
        """_build_dlc_model returns an RTMDetModel with qcs6490/NHWC config for RTM_DET."""
        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import _build_dlc_model
        from moment_to_action.models import ModelID
        from moment_to_action.models.image.detection.rtmdet._model import RTMDetModel

        model = _build_dlc_model(ModelID.RTM_DET, tmp_path)
        assert isinstance(model, RTMDetModel)
        assert model.input_layout == "NHWC"

    def test_build_dlc_model_detectron2_returns_detectron2_model(self, tmp_path: Path) -> None:
        """_build_dlc_model returns a Detectron2Model with qcs6490/NHWC config."""
        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import _build_dlc_model
        from moment_to_action.models import ModelID
        from moment_to_action.models.image.detection.detectron2._model import Detectron2Model

        model = _build_dlc_model(ModelID.DETECTRON2, tmp_path)
        assert isinstance(model, Detectron2Model)
        assert model.input_layout == "NHWC"

    def test_build_dlc_model_unknown_raises(self, tmp_path: Path) -> None:
        """_build_dlc_model raises ClickException for model IDs with no factory."""
        import click

        from moment_to_action._cli.commands.cmd_model.cmd_convert_aihub import (
            _build_dlc_model,
        )
        from moment_to_action.models import ModelID

        with pytest.raises(click.ClickException, match="No DLC model factory"):
            _build_dlc_model(ModelID.MOBILENET_V2, tmp_path)
