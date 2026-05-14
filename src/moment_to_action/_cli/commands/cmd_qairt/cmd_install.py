"""Install QAIRT SDK command."""

from __future__ import annotations

import json

import rich_click as click
from rich.console import Console

from moment_to_action.config import save_config
from moment_to_action.qairt import QairtSDKManager
from moment_to_action.utils.cli import GlobalData, pass_global_data
from moment_to_action.utils.schemas import update_frozen


@click.command()
@click.option("--json", "json_output", is_flag=True, help="Output result as JSON.")
@pass_global_data
def install(data: GlobalData, *, json_output: bool) -> None:
    """Install the QAIRT SDK.

    Downloads and extracts the configured SDK version into the application data
    directory. The installed path is saved to config so other commands can find it.

    Checks required system packages before downloading and runs a post-install
    verification check after a successful install.
    """
    console = Console(stderr=True)
    mgr = QairtSDKManager.from_app_config(data.config, data.path_manager)

    missing = mgr.check_system_deps()
    if missing:
        console.print(
            f"[red]Missing system packages:[/red] {', '.join(missing)}\n"
            "Install them before running 'm2a qairt install'."
        )
        raise click.Abort

    try:
        sdk_path = mgr.install(stream=not json_output)
    except RuntimeError as e:
        error_msg = str(e)
        if "already installed" in error_msg:
            if not click.confirm(f"{error_msg}\n\nReinstall?"):
                raise click.Abort from e
            mgr.clean()
            sdk_path = mgr.install(stream=not json_output)
        else:
            raise click.ClickException(error_msg) from e

    updated = update_frozen(data.config, qairt_sdk_path=sdk_path)
    save_config(updated, data.path_manager.app_config_file)
    data.config = updated

    for issue in mgr.verify(stream=not json_output):
        data.log.warning(issue)

    if json_output:
        click.echo(json.dumps({"path": str(sdk_path), "version": mgr.installed_version}))
    else:
        Console().print(
            f"[green]✓ Installed QAIRT SDK {mgr.installed_version}[/green] → {sdk_path}"
        )
