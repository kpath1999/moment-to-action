"""Clear cache command."""

from __future__ import annotations

import json

import rich_click as click
from rich.console import Console

from moment_to_action.models import ModelManager
from moment_to_action.utils.cli import format_size


@click.command()
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
@click.option(
    "--force",
    is_flag=True,
    help="Skip confirmation prompt and clear cache immediately.",
)
@click.pass_context
def clear(ctx: click.Context, *, json_output: bool, force: bool) -> None:
    """Clear all downloaded models from the cache.

    This command removes all cached (non-vendored) models from the local cache
    directory. Vendored models that ship with the package are not affected.

    By default, a confirmation prompt is shown before clearing. Use --force to
    skip the confirmation.

    Use --json to get machine-readable output in JSON format.
    """
    manager = ModelManager()

    # Confirmation prompt (skip if --json or --force)
    if not json_output and not force:
        console = Console()
        try:
            confirmed = console.input(
                "[yellow]This will remove all downloaded models. Continue? \\[y/N][/yellow] "
            )
            if confirmed.lower() not in ("y", "yes"):
                console.print("[cyan]Cache clear cancelled.[/cyan]")
                ctx.exit(0)
        except EOFError:
            # Non-interactive mode, skip confirmation
            pass

    # Clear cache
    total_bytes, removed_ids = manager.clear_cache()

    # Get cache directory
    cache_dir = manager.cache_dir

    if json_output:
        output = {
            "status": "success",
            "total_bytes_freed": total_bytes,
            "models_removed": [model_id.value for model_id in removed_ids],
            "cache_dir": str(cache_dir),
        }
        click.echo(json.dumps(output))
    else:
        # Rich formatted output
        console = Console()

        if not removed_ids:
            console.print("[cyan]Cache is already empty.[/cyan]")
            return

        console.print("[green]✓ Cache cleared successfully[/green]")
        console.print(f"Removed [bold]{len(removed_ids)}[/bold] model(s):")

        # Display each removed model with its freed size
        for model_id in removed_ids:
            size_str = format_size(total_bytes // len(removed_ids))
            console.print(f"  - {model_id.value}: freed {size_str}")

        console.print()
        console.print(f"Total freed: [bold]{format_size(total_bytes)}[/bold]")
