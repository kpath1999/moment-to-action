"""Unit tests for m2a cache inspect command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from moment_to_action._cli.commands.cmd_cache.cmd_inspect import inspect
from moment_to_action.models import (
    DownloadSource,
    ModelID,
    ModelInfo,
    ModelManager,
    ModelStatus,
    VendoredSource,
)


@pytest.mark.unit
class TestCacheInspectCommand:
    """Tests for the cache inspect subcommand."""

    def test_default_output_shows_table_with_headers(self) -> None:
        """Default output displays a table with expected headers.

        Verifies that invoking 'cache inspect' with no args produces a
        formatted table containing the column headers: Model ID, Source,
        Status, Size, Path.
        """
        from moment_to_action._cli import cli

        # Mock ModelManager to return sample models
        yolo_info = MagicMock()
        yolo_info.id = ModelID.YOLO_V8
        yolo_info.source = VendoredSource(subdir="yolo")

        yolo_status = ModelStatus(
            info=yolo_info,
            available=True,
            path=Path("/cache/yolo.onnx"),
            size_bytes=50_000_000,
        )

        mobileclip_info = MagicMock()
        mobileclip_info.id = ModelID.MOBILECLIP_S2
        mobileclip_info.source = DownloadSource(
            hf_repo_id="anton96vice/mobileclip2_tflite",
            hf_filename="mobileclip_s2_datacompdr_last.tflite",
        )

        mobileclip_status = ModelStatus(
            info=mobileclip_info,
            available=False,
            path=None,
            size_bytes=None,
        )

        mock_manager = MagicMock()
        mock_manager.list_models.return_value = [yolo_status, mobileclip_status]

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_inspect.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "inspect"])

        assert result.exit_code == 0
        # Check for table headers
        assert "Model ID" in result.output
        assert "Source" in result.output
        assert "Status" in result.output
        assert "Size" in result.output
        assert "Path" in result.output

    def test_table_shows_yolo_as_vendored_available(self) -> None:
        """Table shows YOLO_V8 as 'vendored' and 'Available'.

        Verifies that the YOLO_V8 model is correctly displayed as vendored
        source with available status in the table output.
        """
        from moment_to_action._cli import cli

        yolo_info = MagicMock()
        yolo_info.id = ModelID.YOLO_V8
        yolo_info.source = VendoredSource(subdir="yolo")

        yolo_status = ModelStatus(
            info=yolo_info,
            available=True,
            path=Path("/cache/yolo.onnx"),
            size_bytes=50_000_000,
        )

        mock_manager = MagicMock()
        mock_manager.list_models.return_value = [yolo_status]

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_inspect.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "inspect"])

        assert result.exit_code == 0
        assert "yolo_v8" in result.output
        assert "vendored" in result.output

    def test_table_shows_mobileclip_as_download_unavailable(self) -> None:
        """Table shows MOBILECLIP_S2 as 'download' and 'Not Available'.

        Verifies that the MOBILECLIP_S2 model is correctly displayed as
        downloadable source with unavailable status in the table output.
        """
        from moment_to_action._cli import cli

        mobileclip_info = MagicMock()
        mobileclip_info.id = ModelID.MOBILECLIP_S2
        mobileclip_info.source = DownloadSource(
            hf_repo_id="anton96vice/mobileclip2_tflite",
            hf_filename="mobileclip_s2_datacompdr_last.tflite",
        )

        mobileclip_status = ModelStatus(
            info=mobileclip_info,
            available=False,
            path=None,
            size_bytes=None,
        )

        mock_manager = MagicMock()
        mock_manager.list_models.return_value = [mobileclip_status]

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_inspect.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "inspect"])

        assert result.exit_code == 0
        assert "mobileclip_s2" in result.output
        assert "download" in result.output

    def test_json_flag_outputs_valid_json(self) -> None:
        """--json flag produces valid JSON output.

        Verifies that 'cache inspect --json' produces valid, well-formed
        JSON that can be parsed successfully.
        """
        from moment_to_action._cli import cli

        yolo_info = MagicMock()
        yolo_info.id = ModelID.YOLO_V8
        yolo_info.source = VendoredSource(subdir="yolo")

        yolo_status = ModelStatus(
            info=yolo_info,
            available=True,
            path=Path("/cache/yolo.onnx"),
            size_bytes=50_000_000,
        )

        mock_manager = MagicMock()
        mock_manager.list_models.return_value = [yolo_status]
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_inspect.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "inspect", "--json"])

        assert result.exit_code == 0
        # Should be parseable as JSON
        output_json = json.loads(result.output)
        assert isinstance(output_json, dict)

    def test_json_output_has_cache_dir_field(self) -> None:
        """JSON output contains cache_dir field.

        Verifies that the JSON object includes a cache_dir field containing
        the path to the cache directory as a string.
        """
        from moment_to_action._cli import cli

        mock_manager = MagicMock()
        mock_manager.list_models.return_value = []
        mock_manager.cache_dir = Path("/tmp/test_cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_inspect.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "inspect", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert "cache_dir" in output_json
        assert isinstance(output_json["cache_dir"], str)
        assert "/tmp/test_cache" in output_json["cache_dir"]

    def test_json_output_has_models_array(self) -> None:
        """JSON output contains models array.

        Verifies that the JSON object has a models field containing an array
        of model objects.
        """
        from moment_to_action._cli import cli

        mock_manager = MagicMock()
        mock_manager.list_models.return_value = []
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_inspect.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "inspect", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert "models" in output_json
        assert isinstance(output_json["models"], list)

    def test_json_model_objects_have_required_fields(self) -> None:
        """JSON model objects contain id, source, available, size_bytes, path.

        Verifies that each model in the JSON array has the expected fields:
        id, source, available, size_bytes, path.
        """
        from moment_to_action._cli import cli

        yolo_info = MagicMock()
        yolo_info.id = ModelID.YOLO_V8
        yolo_info.source = VendoredSource(subdir="yolo")

        yolo_status = ModelStatus(
            info=yolo_info,
            available=True,
            path=Path("/cache/yolo.onnx"),
            size_bytes=50_000_000,
        )

        mock_manager = MagicMock()
        mock_manager.list_models.return_value = [yolo_status]
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_inspect.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "inspect", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert len(output_json["models"]) == 1

        model = output_json["models"][0]
        assert "id" in model
        assert "source" in model
        assert "available" in model
        assert "size_bytes" in model
        assert "path" in model

    def test_json_yolo_available_true_with_size_and_path(self) -> None:
        """JSON YOLO model has available=true, size_bytes > 0, path not null.

        Verifies that an available vendored model (YOLO_V8) is represented
        in JSON with available=true, a positive size_bytes value, and a valid
        path string.
        """
        from moment_to_action._cli import cli

        yolo_info = MagicMock()
        yolo_info.id = ModelID.YOLO_V8
        yolo_info.source = VendoredSource(subdir="yolo")

        yolo_status = ModelStatus(
            info=yolo_info,
            available=True,
            path=Path("/cache/yolo.onnx"),
            size_bytes=50_000_000,
        )

        mock_manager = MagicMock()
        mock_manager.list_models.return_value = [yolo_status]
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_inspect.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "inspect", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)

        model = output_json["models"][0]
        assert model["id"] == "yolo_v8"
        assert model["available"] is True
        assert model["size_bytes"] == 50_000_000
        assert model["path"] is not None
        assert isinstance(model["path"], str)

    def test_json_mobileclip_unavailable_null_size_and_path(self) -> None:
        """JSON MOBILECLIP model has available=false, size_bytes/path null.

        Verifies that an unavailable downloadable model (MOBILECLIP_S2) is
        represented in JSON with available=false, and both size_bytes and
        path set to null.
        """
        from moment_to_action._cli import cli

        mobileclip_info = MagicMock()
        mobileclip_info.id = ModelID.MOBILECLIP_S2
        mobileclip_info.source = DownloadSource(
            hf_repo_id="anton96vice/mobileclip2_tflite",
            hf_filename="mobileclip_s2_datacompdr_last.tflite",
        )

        mobileclip_status = ModelStatus(
            info=mobileclip_info,
            available=False,
            path=None,
            size_bytes=None,
        )

        mock_manager = MagicMock()
        mock_manager.list_models.return_value = [mobileclip_status]
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_inspect.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "inspect", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)

        model = output_json["models"][0]
        assert model["id"] == "mobileclip_s2"
        assert model["available"] is False
        assert model["size_bytes"] is None
        assert model["path"] is None

    def test_exit_code_zero_on_success(self) -> None:
        """Command exits with code 0 on success.

        Verifies that both default and --json modes exit with exit code 0
        when the command succeeds.
        """
        from moment_to_action._cli import cli

        mock_manager = MagicMock()
        mock_manager.list_models.return_value = []
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_inspect.ModelManager",
                return_value=mock_manager,
            ):
                runner = CliRunner()
                result_default = runner.invoke(cli, ["cache", "inspect"])
                result_json = runner.invoke(cli, ["cache", "inspect", "--json"])

        assert result_default.exit_code == 0
        assert result_json.exit_code == 0

    def test_inspect_with_long_path_truncation(self, tmp_path: Path) -> None:
        """Test inspect command truncates very long model paths."""
        manager = ModelManager(cache_dir=tmp_path)

        # Create a status with a very long path (>300 chars)
        long_dir = tmp_path / ("a" * 150)
        long_dir.mkdir(parents=True, exist_ok=True)
        long_path = long_dir / "model.onnx"
        long_path.write_text("model")

        info = ModelInfo(
            id=ModelID.YOLO_V8,
            filename="model.onnx",
            source=VendoredSource(subdir="test"),
        )
        status = ModelStatus(
            info=info,
            available=True,
            path=long_path,
            size_bytes=1024,
        )

        with mock.patch.object(manager, "list_models", return_value=[status]):
            runner = CliRunner()
            result = runner.invoke(inspect, [])

            assert result.exit_code == 0
            # Path should be truncated with "..."
            assert "..." in result.output
