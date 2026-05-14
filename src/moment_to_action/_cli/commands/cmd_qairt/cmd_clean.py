"""Remove QAIRT SDK command."""

from __future__ import annotations

import json

import rich_click as click
from rich.console import Console

from moment_to_action.config import save_config
from moment_to_action.qairt import QairtSDKManager
from moment_to_action.utils.cli import GlobalData, pass_global_data
from moment_to_action.utils.schemas import update_frozen


@click.command()
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
@click.option("--json", "json_output", is_flag=True, help="Output result as JSON.")
@pass_global_data
def clean(data: GlobalData, *, force: bool, json_output: bool) -> None:
    """Remove the installed QAIRT SDK.

    Deletes the SDK directory and clears the configured path from the
    application config. Use --force to skip the confirmation prompt.
    """
    mgr = QairtSDKManager.from_app_config(data.config, data.path_manager)

    if not force and not json_output:
        console = Console()
        try:
            confirmed = console.input(
                "[yellow]This will remove the QAIRT SDK directory. Continue? \\[y/N][/yellow] "
            )
            if confirmed.lower() not in ("y", "yes"):
                console.print("[cyan]Cancelled.[/cyan]")
                return
        except EOFError:
            pass

    try:
        removed = mgr.clean()
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e

    updated = update_frozen(data.config, qairt_sdk_path=None)
    save_config(updated, data.path_manager.app_config_file)
    data.config = updated

    if json_output:
        click.echo(json.dumps({"removed": str(removed)}))
    else:
        Console().print(f"[green]✓ Removed QAIRT SDK[/green] {removed}")
