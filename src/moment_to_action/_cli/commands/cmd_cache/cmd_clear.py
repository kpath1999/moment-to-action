"""Clear cache command."""

from __future__ import annotations

import json

import rich_click as click
from rich.console import Console

from moment_to_action.utils.cli import GlobalData, pass_global_data


@click.command()
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
@click.option(
    "--force",
    is_flag=True,
    help="Skip confirmation prompt and clear cache immediately.",
)
@pass_global_data
def clear(data: GlobalData, *, json_output: bool, force: bool) -> None:
    """Clear the application cache.

    Removes everything in the cache directory (cached models and any other
    cache contents).

    By default, a confirmation prompt is shown before clearing. Use --force to
    skip the confirmation.

    Use --json to get machine-readable output in JSON format.
    """
    if not json_output and not force:
        console = Console()
        try:
            confirmed = console.input(
                "[yellow]This will remove all cached files. Continue? \\[y/N][/yellow] "
            )
            if confirmed.lower() not in ("y", "yes"):
                console.print("[cyan]Cache clear cancelled.[/cyan]")
                return
        except EOFError:
            pass

    info = data.path_manager.cache.clear_cache()

    if json_output:
        click.echo(json.dumps(info.to_json(), indent=2))
        return

    console = Console()
    if info.total_size_bytes == 0:
        console.print("[cyan]Cache is already empty.[/cyan]")
        return

    console.print("[green]✓ Cache cleared successfully[/green]")
    console.print(info.to_rich_table())
