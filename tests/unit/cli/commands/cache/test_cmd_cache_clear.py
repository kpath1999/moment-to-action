"""Unit tests for m2a cache clear command."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from moment_to_action.config import AppConfig
from moment_to_action.paths._cache._manager import CacheInfo
from moment_to_action.paths._cache._models import ModelCacheContents

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def _make_cache_info(total_bytes: int = 0, root_contents: Iterable[Path] = ()) -> CacheInfo:
    """Return a real CacheInfo with the given total size."""
    return CacheInfo(
        total_size_bytes=total_bytes,
        root_contents=list(root_contents),
        models_info=ModelCacheContents(
            total_size_bytes=total_bytes,
            models={},
            other=[],
        ),
    )


def _patched_path_manager(info: CacheInfo) -> MagicMock:
    """Return a MagicMock standing in for PathManager(), with cache.clear_cache → info."""
    mock_path_man = MagicMock()
    mock_path_man.cache.clear_cache.return_value = info
    return mock_path_man


@pytest.mark.unit
class TestCacheClearCommand:
    """Tests for the cache clear subcommand."""

    def test_clear_with_force_exits_zero(self) -> None:
        """--force flag skips confirmation and exits with code 0."""
        from moment_to_action._cli import cli

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.PathManager",
                return_value=_patched_path_manager(_make_cache_info(0)),
            ):
                with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    result = CliRunner().invoke(cli, ["cache", "clear", "--force"])

        assert result.exit_code == 0

    def test_clear_empty_cache_shows_message(self) -> None:
        """Clearing empty cache shows an appropriate message."""
        from moment_to_action._cli import cli

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.PathManager",
                return_value=_patched_path_manager(_make_cache_info(0)),
            ):
                with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    result = CliRunner().invoke(cli, ["cache", "clear", "--force"])

        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_clear_with_bytes_freed_shows_success_message(self) -> None:
        """Non-zero bytes freed shows a success message."""
        from moment_to_action._cli import cli

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.PathManager",
                return_value=_patched_path_manager(_make_cache_info(100_000_000)),
            ):
                with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    result = CliRunner().invoke(cli, ["cache", "clear", "--force"])

        assert result.exit_code == 0
        assert "cleared" in result.output.lower()

    def test_json_flag_outputs_valid_json(self) -> None:
        """--json flag produces valid JSON."""
        from moment_to_action._cli import cli

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.PathManager",
                return_value=_patched_path_manager(_make_cache_info(100_000_000)),
            ):
                with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    result = CliRunner().invoke(cli, ["cache", "clear", "--force", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert isinstance(output_json, dict)

    def test_json_output_uses_cache_info_to_json(self) -> None:
        """JSON output matches CacheInfo.to_json() shape."""
        from moment_to_action._cli import cli

        freed = 100_000_000
        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.PathManager",
                return_value=_patched_path_manager(_make_cache_info(freed)),
            ):
                with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    result = CliRunner().invoke(cli, ["cache", "clear", "--force", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert output_json["total_size_bytes"] == freed
        assert "root_contents" in output_json
        assert "models_info" in output_json

    def test_confirmation_no_response_cancels_clear(self) -> None:
        """Confirmation 'n' cancels the clear (and clear_cache is not called)."""
        from moment_to_action._cli import cli

        mock_path_man = _patched_path_manager(_make_cache_info(0))
        with patch("moment_to_action._cli.init_logging"):
            with patch("moment_to_action._cli.PathManager", return_value=mock_path_man):
                with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    result = CliRunner().invoke(cli, ["cache", "clear"], input="n\n")

        assert result.exit_code == 0
        assert "cancelled" in result.output.lower()
        mock_path_man.cache.clear_cache.assert_not_called()

    def test_confirmation_yes_response_clears_cache(self) -> None:
        """Confirmation 'y' clears the cache."""
        from moment_to_action._cli import cli

        mock_path_man = _patched_path_manager(_make_cache_info(100_000_000))
        with patch("moment_to_action._cli.init_logging"):
            with patch("moment_to_action._cli.PathManager", return_value=mock_path_man):
                with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    result = CliRunner().invoke(cli, ["cache", "clear"], input="y\n")

        assert result.exit_code == 0
        mock_path_man.cache.clear_cache.assert_called_once()

    def test_json_flag_skips_confirmation_prompt(self) -> None:
        """--json flag skips the confirmation prompt."""
        from moment_to_action._cli import cli

        mock_path_man = _patched_path_manager(_make_cache_info(0))
        with patch("moment_to_action._cli.init_logging"):
            with patch("moment_to_action._cli.PathManager", return_value=mock_path_man):
                with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    result = CliRunner().invoke(cli, ["cache", "clear", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert isinstance(output_json, dict)
        mock_path_man.cache.clear_cache.assert_called_once()

    def test_eoferror_in_non_interactive_proceeds_without_confirmation(self) -> None:
        """Clear handles EOFError in non-interactive mode by proceeding."""
        from moment_to_action._cli import cli

        mock_path_man = _patched_path_manager(_make_cache_info(0))
        with patch("moment_to_action._cli.init_logging"):
            with patch("moment_to_action._cli.PathManager", return_value=mock_path_man):
                with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    result = CliRunner().invoke(cli, ["cache", "clear"], input="")

        assert result.exit_code == 0
        mock_path_man.cache.clear_cache.assert_called_once()
