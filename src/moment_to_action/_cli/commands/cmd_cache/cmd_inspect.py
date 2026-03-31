"""Inspect model cache command."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import rich
import rich_click as click
from rich.console import Console
from rich.table import Table

from moment_to_action.models import (
    DownloadSource,
    ModelManager,
    TransformersSource,
    VendoredSource,
)
from moment_to_action.utils.cli import format_size

if TYPE_CHECKING:
    from moment_to_action.models import ModelStatus

_MAX_PATH_LEN = rich.get_console().size.width - 60
_TRUNCATED_PATH_LEN = _MAX_PATH_LEN - 3


@click.command()
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
def inspect(*, json_output: bool) -> None:
    """Inspect the model cache.

    Displays the status of all known models, including which are available
    locally, their sizes, and where they are stored.
    """
    manager = ModelManager()
    statuses = manager.list_models()

    if json_output:
        _output_json(manager, statuses)
    else:
        _output_table(statuses)


def _output_json(manager: ModelManager, statuses: list[ModelStatus]) -> None:
    """Output model cache status as JSON.

    Args:
        manager: The ModelManager instance (for cache dir access).
        statuses: List of ModelStatus objects.
    """
    output: dict = {
        "cache_dir": str(manager.cache_dir),
        "models": [],
    }

    for status in statuses:
        # Determine source string
        match status.info.source:
            case VendoredSource():
                source_str = "vendored"
            case DownloadSource():
                source_str = "download"
            case TransformersSource():
                source_str = "transformers"

        model_data = {
            "id": status.info.id.value,
            "source": source_str,
            "available": status.available,
            "size_bytes": status.size_bytes,
            "path": str(status.path) if status.path else None,
        }
        output["models"].append(model_data)

    click.echo(json.dumps(output, indent=2))


def _output_table(statuses: list[ModelStatus]) -> None:
    """Output model cache status as a formatted table.

    Args:
        statuses: List of ModelStatus objects.
    """
    console = Console()
    table = Table(title="Model Cache Status")

    table.add_column("Model ID", style="cyan")
    table.add_column("Source", style="blue")
    table.add_column("Status", style="magenta")
    table.add_column("Size")
    table.add_column("Path")

    for status in statuses:
        # Determine source string
        match status.info.source:
            case VendoredSource():
                source_str = "vendored"
            case DownloadSource():
                source_str = "download"
            case TransformersSource():
                source_str = "transformers"

        # Format status
        status_str = "[green]Available[/green]" if status.available else "[red]Not Available[/red]"

        # Format size
        size_str = format_size(status.size_bytes) if status.size_bytes else "-"

        # Format path (truncate if very long)
        path_str = str(status.path) if status.path else "-"
        if len(path_str) > _MAX_PATH_LEN:
            path_str = "..." + path_str[-_TRUNCATED_PATH_LEN:]

        table.add_row(
            status.info.id.value,
            source_str,
            status_str,
            size_str,
            path_str,
        )

    console.print(table)
