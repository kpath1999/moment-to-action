"""QAIRT SDK info command."""

from __future__ import annotations

import json

import rich_click as click
from rich.console import Console
from rich.table import Table

from moment_to_action.qairt import QairtSDKManager
from moment_to_action.utils.cli import GlobalData, pass_global_data


@click.command()
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
@pass_global_data
def info(data: GlobalData, *, json_output: bool) -> None:
    """Show QAIRT SDK information.

    Displays the configured version, installed path, availability, and the
    full version string (including build number) derived from the install path.
    """
    mgr = QairtSDKManager.from_app_config(data.config, data.path_manager)

    if json_output:
        click.echo(
            json.dumps(
                {
                    "configured_version": mgr.configured_version,
                    "installed_version": mgr.installed_version,
                    "path": str(mgr.path) if mgr.path else None,
                    "available": mgr.is_available,
                },
                indent=2,
            )
        )
        return

    table = Table(title="QAIRT SDK")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Configured version", mgr.configured_version)
    table.add_row("Installed version", mgr.installed_version or "[dim]not installed[/dim]")
    table.add_row("Path", str(mgr.path) if mgr.path else "[dim]not installed[/dim]")
    table.add_row("Available", "[green]yes[/green]" if mgr.is_available else "[red]no[/red]")
    Console().print(table)
