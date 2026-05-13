"""Inspect cache command."""

from __future__ import annotations

import json

import rich_click as click
from rich.console import Console

from moment_to_action.utils.cli import GlobalData, pass_global_data


@click.command()
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
@pass_global_data
def inspect(data: GlobalData, *, json_output: bool) -> None:
    """Inspect the application cache.

    Displays a summary of the cache contents (size, sub-caches, unexpected
    files). Use --json to get machine-readable output in JSON format.
    """
    info = data.path_manager.cache.inspect_cache()

    if json_output:
        click.echo(json.dumps(info.to_json(), indent=2))
        return

    Console().print(info.to_rich_table())
