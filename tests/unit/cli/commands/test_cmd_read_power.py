"""Unit tests for _cli/commands/cmd_read_power.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


@pytest.mark.unit
class TestReadPowerCommand:
    """Tests for the read-power subcommand."""

    def test_outputs_power_and_utilization(self) -> None:
        """read-power echoes power_mw and utilization_pct for the chosen device."""
        from moment_to_action._cli import cli

        mock_sample = MagicMock()
        mock_sample.power_mw = 1500
        mock_sample.utilization_pct = 42

        mock_pwr_mon = MagicMock()
        mock_pwr_mon.sample.return_value = mock_sample

        mock_backend = MagicMock()
        mock_backend.power_monitor = mock_pwr_mon

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=mock_backend,
            ):
                result = CliRunner().invoke(cli, ["read-power", "CPU"])

        assert result.exit_code == 0
        assert "1500" in result.output
        assert "42" in result.output

    def test_default_output_contains_power_in_mw(self) -> None:
        """Default output displays power measurement in milliwatts."""
        from moment_to_action._cli import cli

        mock_sample = MagicMock()
        mock_sample.power_mw = 2500.5
        mock_sample.utilization_pct = 10

        mock_pwr_mon = MagicMock()
        mock_pwr_mon.sample.return_value = mock_sample

        mock_backend = MagicMock()
        mock_backend.power_monitor = mock_pwr_mon

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=mock_backend,
            ):
                result = CliRunner().invoke(cli, ["read-power", "CPU"])

        assert result.exit_code == 0
        assert "mW" in result.output

    def test_default_output_contains_utilization_percentage(self) -> None:
        """Default output displays utilization as a percentage."""
        from moment_to_action._cli import cli

        mock_sample = MagicMock()
        mock_sample.power_mw = 1500
        mock_sample.utilization_pct = 67.5

        mock_pwr_mon = MagicMock()
        mock_pwr_mon.sample.return_value = mock_sample

        mock_backend = MagicMock()
        mock_backend.power_monitor = mock_pwr_mon

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=mock_backend,
            ):
                result = CliRunner().invoke(cli, ["read-power", "CPU"])

        assert result.exit_code == 0
        assert "67.5" in result.output or "67" in result.output
        assert "%" in result.output

    def test_default_output_contains_device_name(self) -> None:
        """Default output displays the queried device name."""
        from moment_to_action._cli import cli

        mock_sample = MagicMock()
        mock_sample.power_mw = 1500
        mock_sample.utilization_pct = 42

        mock_pwr_mon = MagicMock()
        mock_pwr_mon.sample.return_value = mock_sample

        mock_backend = MagicMock()
        mock_backend.power_monitor = mock_pwr_mon

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=mock_backend,
            ):
                result = CliRunner().invoke(cli, ["read-power", "CPU"])

        assert result.exit_code == 0
        assert "CPU" in result.output

    def test_json_flag_outputs_valid_json(self) -> None:
        """--json flag produces valid JSON output."""
        from moment_to_action._cli import cli

        mock_sample = MagicMock()
        mock_sample.power_mw = 1500.5
        mock_sample.utilization_pct = 42.0
        mock_sample.timestamp = 1234567890.5

        mock_pwr_mon = MagicMock()
        mock_pwr_mon.sample.return_value = mock_sample

        mock_backend = MagicMock()
        mock_backend.power_monitor = mock_pwr_mon

        asdict_result = {
            "timestamp": 1234567890.5,
            "compute_unit": "CPU",
            "power_mw": 1500.5,
            "utilization_pct": 42.0,
        }
        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=mock_backend,
            ):
                with patch(
                    "moment_to_action._cli.commands.cmd_read_power.attrs.asdict",
                    return_value=asdict_result,
                ):
                    result = CliRunner().invoke(cli, ["read-power", "CPU", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert isinstance(output_json, dict)

    def test_json_output_has_compute_unit_field(self) -> None:
        """JSON output has compute_unit field."""
        from moment_to_action._cli import cli

        mock_sample = MagicMock()

        mock_pwr_mon = MagicMock()
        mock_pwr_mon.sample.return_value = mock_sample

        mock_backend = MagicMock()
        mock_backend.power_monitor = mock_pwr_mon

        asdict_result = {
            "timestamp": 1234567890.5,
            "compute_unit": "CPU",
            "power_mw": 1500,
            "utilization_pct": 42,
        }
        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=mock_backend,
            ):
                with patch(
                    "moment_to_action._cli.commands.cmd_read_power.attrs.asdict",
                    return_value=asdict_result,
                ):
                    result = CliRunner().invoke(cli, ["read-power", "CPU", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert "compute_unit" in output_json

    def test_json_output_has_power_mw_field(self) -> None:
        """JSON output has power_mw field."""
        from moment_to_action._cli import cli

        mock_sample = MagicMock()

        mock_pwr_mon = MagicMock()
        mock_pwr_mon.sample.return_value = mock_sample

        mock_backend = MagicMock()
        mock_backend.power_monitor = mock_pwr_mon

        asdict_result = {
            "timestamp": 1234567890.5,
            "compute_unit": "CPU",
            "power_mw": 2345.67,
            "utilization_pct": 50,
        }
        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=mock_backend,
            ):
                with patch(
                    "moment_to_action._cli.commands.cmd_read_power.attrs.asdict",
                    return_value=asdict_result,
                ):
                    result = CliRunner().invoke(cli, ["read-power", "CPU", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert "power_mw" in output_json
        assert output_json["power_mw"] == 2345.67

    def test_json_output_has_utilization_pct_field(self) -> None:
        """JSON output has utilization_pct field."""
        from moment_to_action._cli import cli

        mock_sample = MagicMock()

        mock_pwr_mon = MagicMock()
        mock_pwr_mon.sample.return_value = mock_sample

        mock_backend = MagicMock()
        mock_backend.power_monitor = mock_pwr_mon

        asdict_result = {
            "timestamp": 1234567890.5,
            "compute_unit": "CPU",
            "power_mw": 1500,
            "utilization_pct": 75.5,
        }
        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=mock_backend,
            ):
                with patch(
                    "moment_to_action._cli.commands.cmd_read_power.attrs.asdict",
                    return_value=asdict_result,
                ):
                    result = CliRunner().invoke(cli, ["read-power", "CPU", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert "utilization_pct" in output_json
        assert output_json["utilization_pct"] == 75.5

    def test_json_output_has_timestamp_field(self) -> None:
        """JSON output has timestamp field."""
        from moment_to_action._cli import cli

        test_timestamp = 1704067200.5
        mock_sample = MagicMock()

        mock_pwr_mon = MagicMock()
        mock_pwr_mon.sample.return_value = mock_sample

        mock_backend = MagicMock()
        mock_backend.power_monitor = mock_pwr_mon

        asdict_result = {
            "timestamp": test_timestamp,
            "compute_unit": "CPU",
            "power_mw": 1500,
            "utilization_pct": 42,
        }
        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=mock_backend,
            ):
                with patch(
                    "moment_to_action._cli.commands.cmd_read_power.attrs.asdict",
                    return_value=asdict_result,
                ):
                    result = CliRunner().invoke(cli, ["read-power", "CPU", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert "timestamp" in output_json
        assert output_json["timestamp"] == test_timestamp

    @pytest.mark.parametrize("device", ["CPU", "NPU", "GPU", "DSP"])
    def test_all_compute_units_succeed(self, device: str) -> None:
        """All compute units (CPU, NPU, GPU, DSP) can be queried successfully."""
        from moment_to_action._cli import cli

        mock_sample = MagicMock()
        mock_sample.power_mw = 1000
        mock_sample.utilization_pct = 50

        mock_pwr_mon = MagicMock()
        mock_pwr_mon.sample.return_value = mock_sample

        mock_backend = MagicMock()
        mock_backend.power_monitor = mock_pwr_mon

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=mock_backend,
            ):
                result = CliRunner().invoke(cli, ["read-power", device])

        assert result.exit_code == 0
        assert "1000" in result.output or "50" in result.output

    @pytest.mark.parametrize("device", ["CPU", "NPU", "GPU", "DSP"])
    def test_all_compute_units_json_output(self, device: str) -> None:
        """JSON output works for all compute units."""
        from moment_to_action._cli import cli

        mock_sample = MagicMock()

        mock_pwr_mon = MagicMock()
        mock_pwr_mon.sample.return_value = mock_sample

        mock_backend = MagicMock()
        mock_backend.power_monitor = mock_pwr_mon

        asdict_result = {
            "timestamp": 1234567890.0,
            "compute_unit": device,
            "power_mw": 1000,
            "utilization_pct": 50,
        }
        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=mock_backend,
            ):
                with patch(
                    "moment_to_action._cli.commands.cmd_read_power.attrs.asdict",
                    return_value=asdict_result,
                ):
                    result = CliRunner().invoke(cli, ["read-power", device, "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert "compute_unit" in output_json
        assert output_json["compute_unit"] == device
        assert "power_mw" in output_json
        assert "utilization_pct" in output_json
        assert "timestamp" in output_json

    def test_invalid_device_fails(self) -> None:
        """Invalid device name causes command to fail."""
        from moment_to_action._cli import cli

        with patch("moment_to_action._cli.init_logging"):
            result = CliRunner().invoke(cli, ["read-power", "INVALID"])

        assert result.exit_code != 0

    def test_exit_code_zero_on_success(self) -> None:
        """Command exits with code 0 on success."""
        from moment_to_action._cli import cli

        mock_sample = MagicMock()
        mock_sample.power_mw = 1500
        mock_sample.utilization_pct = 42

        mock_pwr_mon = MagicMock()
        mock_pwr_mon.sample.return_value = mock_sample

        mock_backend = MagicMock()
        mock_backend.power_monitor = mock_pwr_mon

        asdict_result = {
            "timestamp": 1234567890.0,
            "compute_unit": "CPU",
            "power_mw": 1500,
            "utilization_pct": 42,
        }
        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=mock_backend,
            ):
                with patch(
                    "moment_to_action._cli.commands.cmd_read_power.attrs.asdict",
                    return_value=asdict_result,
                ):
                    runner = CliRunner()
                    result_default = runner.invoke(cli, ["read-power", "CPU"])
                    result_json = runner.invoke(cli, ["read-power", "CPU", "--json"])

        assert result_default.exit_code == 0
        assert result_json.exit_code == 0

    def test_alias_rdpwr_works(self) -> None:
        """Alias 'rdpwr' works as shorthand for 'read-power'."""
        from moment_to_action._cli import cli

        mock_sample = MagicMock()
        mock_sample.power_mw = 1500
        mock_sample.utilization_pct = 42
        mock_sample.timestamp = 1234567890.0

        mock_pwr_mon = MagicMock()
        mock_pwr_mon.sample.return_value = mock_sample

        mock_backend = MagicMock()
        mock_backend.power_monitor = mock_pwr_mon

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_read_power.ComputeBackend",
                return_value=mock_backend,
            ):
                result = CliRunner().invoke(cli, ["rdpwr", "CPU"])

        assert result.exit_code == 0
        assert "1500" in result.output
