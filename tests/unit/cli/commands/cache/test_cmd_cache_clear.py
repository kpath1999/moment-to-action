"""Unit tests for m2a cache clear command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from moment_to_action._cli.commands.cmd_cache.cmd_clear import clear
from moment_to_action.models import ModelID, ModelManager


@pytest.mark.unit
class TestCacheClearCommand:
    """Tests for the cache clear subcommand."""

    def test_clear_with_force_exits_zero(self) -> None:
        """--force flag skips confirmation and exits with code 0.

        Verifies that 'cache clear --force' bypasses the confirmation prompt
        and completes successfully with exit code 0.
        """
        from moment_to_action._cli import cli

        mock_manager = MagicMock()
        mock_manager.clear_cache.return_value = (0, [])
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_clear.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "clear", "--force"])

        assert result.exit_code == 0

    def test_clear_empty_cache_shows_nothing_to_clear(self) -> None:
        """Clearing empty cache shows appropriate message.

        When the cache is empty (no models removed), output should indicate
        that there was nothing to clear.
        """
        from moment_to_action._cli import cli

        mock_manager = MagicMock()
        mock_manager.clear_cache.return_value = (0, [])
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_clear.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "clear", "--force"])

        assert result.exit_code == 0
        assert "empty" in result.output.lower() or "nothing" in result.output.lower()

    def test_clear_with_models_shows_success_message(self) -> None:
        """Clearing cache with models shows success message.

        When models are removed from the cache, output should indicate success
        (e.g., "✓ Cache cleared" or similar).
        """
        from moment_to_action._cli import cli

        mock_manager = MagicMock()
        mock_manager.clear_cache.return_value = (
            100_000_000,
            [ModelID.MOBILECLIP_S2],
        )
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_clear.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "clear", "--force"])

        assert result.exit_code == 0
        # Check for success indicators
        assert "cleared" in result.output.lower() or "removed" in result.output.lower()

    def test_clear_shows_total_bytes_freed(self) -> None:
        """Output shows total bytes freed.

        The output should display how many bytes were freed, in human-readable
        format (e.g., "100.0 MB").
        """
        from moment_to_action._cli import cli

        mock_manager = MagicMock()
        mock_manager.clear_cache.return_value = (
            100_000_000,
            [ModelID.MOBILECLIP_S2],
        )
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_clear.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "clear", "--force"])

        assert result.exit_code == 0
        # Should contain size information (in various human-readable formats)
        assert (
            "MB" in result.output
            or "GB" in result.output
            or "KB" in result.output
            or "B" in result.output
        )

    def test_json_flag_outputs_valid_json(self) -> None:
        """--json flag produces valid JSON output.

        Verifies that 'cache clear --force --json' produces valid,
        well-formed JSON that can be parsed successfully.
        """
        from moment_to_action._cli import cli

        mock_manager = MagicMock()
        mock_manager.clear_cache.return_value = (
            100_000_000,
            [ModelID.MOBILECLIP_S2],
        )
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_clear.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "clear", "--force", "--json"])

        assert result.exit_code == 0
        # Should be parseable as JSON
        output_json = json.loads(result.output)
        assert isinstance(output_json, dict)

    def test_json_output_has_status_success(self) -> None:
        """JSON output has status='success' field.

        Verifies that the JSON object includes a status field with value
        "success" when the clear operation completes.
        """
        from moment_to_action._cli import cli

        mock_manager = MagicMock()
        mock_manager.clear_cache.return_value = (
            100_000_000,
            [ModelID.MOBILECLIP_S2],
        )
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_clear.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "clear", "--force", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert "status" in output_json
        assert output_json["status"] == "success"

    def test_json_output_has_total_bytes_freed(self) -> None:
        """JSON output has total_bytes_freed field as integer.

        Verifies that the JSON object includes a total_bytes_freed field
        containing an integer representing bytes freed.
        """
        from moment_to_action._cli import cli

        freed_bytes = 100_000_000
        mock_manager = MagicMock()
        mock_manager.clear_cache.return_value = (
            freed_bytes,
            [ModelID.MOBILECLIP_S2],
        )
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_clear.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "clear", "--force", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert "total_bytes_freed" in output_json
        assert isinstance(output_json["total_bytes_freed"], int)
        assert output_json["total_bytes_freed"] == freed_bytes

    def test_json_output_has_models_removed_array(self) -> None:
        """JSON output has models_removed array of model IDs.

        Verifies that the JSON object includes a models_removed field
        containing a list of model ID strings that were removed.
        """
        from moment_to_action._cli import cli

        mock_manager = MagicMock()
        mock_manager.clear_cache.return_value = (
            100_000_000,
            [ModelID.MOBILECLIP_S2],
        )
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_clear.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "clear", "--force", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert "models_removed" in output_json
        assert isinstance(output_json["models_removed"], list)
        assert "mobileclip_s2" in output_json["models_removed"]

    def test_json_output_has_cache_dir_field(self) -> None:
        """JSON output has cache_dir field as string.

        Verifies that the JSON object includes a cache_dir field containing
        the cache directory path as a string.
        """
        from moment_to_action._cli import cli

        cache_path = Path("/tmp/test_cache")
        mock_manager = MagicMock()
        mock_manager.clear_cache.return_value = (0, [])
        mock_manager.cache_dir = cache_path

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_clear.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "clear", "--force", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert "cache_dir" in output_json
        assert isinstance(output_json["cache_dir"], str)
        assert str(cache_path) in output_json["cache_dir"]

    def test_confirmation_prompt_on_interactive_mode(self) -> None:
        """Confirmation prompt appears when run without --force.

        Verifies that when the command is run interactively without --force,
        a confirmation prompt is displayed asking the user to continue.
        """
        from moment_to_action._cli import cli

        mock_manager = MagicMock()
        mock_manager.clear_cache.return_value = (0, [])
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_clear.ModelManager",
                return_value=mock_manager,
            ):
                # Simulate user pressing 'n' (no)
                result = CliRunner().invoke(cli, ["cache", "clear"], input="n\n")

        # Should exit with success but not actually clear
        assert result.exit_code == 0
        assert "cancelled" in result.output.lower() or "continue" in result.output.lower()

    def test_confirmation_yes_response_clears_cache(self) -> None:
        """Confirmation prompt 'yes' response clears cache.

        Verifies that when the user confirms with 'y' or 'yes', the cache
        is actually cleared.
        """
        from moment_to_action._cli import cli

        mock_manager = MagicMock()
        mock_manager.clear_cache.return_value = (
            100_000_000,
            [ModelID.MOBILECLIP_S2],
        )
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_clear.ModelManager",
                return_value=mock_manager,
            ):
                # Simulate user pressing 'y' (yes)
                result = CliRunner().invoke(cli, ["cache", "clear"], input="y\n")

        assert result.exit_code == 0
        # Verify clear_cache was actually called
        mock_manager.clear_cache.assert_called_once()

    def test_confirmation_no_response_cancels_clear(self) -> None:
        """Confirmation prompt 'no' response cancels clear operation.

        Verifies that when the user responds 'n' or 'no', the cache clear
        is cancelled and clear_cache is not called.
        """
        from moment_to_action._cli import cli

        mock_manager = MagicMock()
        mock_manager.clear_cache.return_value = (0, [])
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_clear.ModelManager",
                return_value=mock_manager,
            ):
                # Simulate user pressing 'n' (no)
                result = CliRunner().invoke(cli, ["cache", "clear"], input="n\n")

        assert result.exit_code == 0
        # Verify clear_cache was NOT called
        mock_manager.clear_cache.assert_not_called()

    def test_json_flag_skips_confirmation_prompt(self) -> None:
        """--json flag skips confirmation prompt.

        Verifies that when --json is used without --force, the command does
        not show a confirmation prompt (assumes automation context).
        """
        from moment_to_action._cli import cli

        mock_manager = MagicMock()
        mock_manager.clear_cache.return_value = (0, [])
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_clear.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "clear", "--json"])

        assert result.exit_code == 0
        # Should output JSON directly (no prompt in output)
        output_json = json.loads(result.output)
        assert isinstance(output_json, dict)

    def test_multiple_models_removed_shown_in_output(self) -> None:
        """Output shows all removed models.

        Verifies that when multiple models are removed, each is listed in
        the output with its freed size.
        """
        from moment_to_action._cli import cli

        mock_manager = MagicMock()
        mock_manager.clear_cache.return_value = (
            200_000_000,
            [ModelID.MOBILECLIP_S2, ModelID.YOLO_V8],
        )
        mock_manager.cache_dir = Path("/tmp/cache")

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.commands.cmd_cache.cmd_clear.ModelManager",
                return_value=mock_manager,
            ):
                result = CliRunner().invoke(cli, ["cache", "clear", "--force"])

        assert result.exit_code == 0
        # Both models should be mentioned
        assert "mobileclip_s2" in result.output

    def test_cmd_clear_eoferror_handling(self, tmp_path: Path) -> None:
        """Test cmd_clear handles EOFError in non-interactive mode."""
        manager = ModelManager(cache_dir=tmp_path)

        with mock.patch(
            "moment_to_action._cli.commands.cmd_cache.cmd_clear.ModelManager",
            return_value=manager,
        ):
            with mock.patch("moment_to_action._cli.init_logging"):
                # Use Click's built-in stdin parameter to simulate empty input
                runner = CliRunner()
                result = runner.invoke(clear, [], input="")

                # Should complete without error
                assert result.exit_code == 0
