"""Config command — get/set application configuration values."""

from __future__ import annotations

import rich_click as click
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from moment_to_action.config import AppConfig, save_config
from moment_to_action.utils.cli import GlobalData, pass_global_data
from moment_to_action.utils.schemas import update_frozen


@click.command()
@click.argument("key", required=False)
@click.argument("value", required=False)
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
@pass_global_data
def config(data: GlobalData, key: str | None, value: str | None, *, json_output: bool) -> None:
    r"""Get or set a configuration value.

    With no arguments, prints the full config. With KEY only, prints the
    current value. With KEY and VALUE, sets the value.

    \b
    Examples:
      m2a config
      m2a config max_workers
      m2a config max_workers 8
    """
    cfg = data.config

    if key is None:
        if json_output:
            click.echo(cfg.model_dump_json(indent=2))
            return
        table = Table(title="Config")
        table.add_column("Key")
        table.add_column("Value")
        for k, v in cfg.model_dump().items():
            table.add_row(k, str(v))
        Console().print(table)
        return

    if key not in AppConfig.model_fields:
        msg = f"Unknown key '{key}'. Valid keys: {', '.join(AppConfig.model_fields)}"
        raise click.BadParameter(msg)

    if value is None:
        click.echo(getattr(cfg, key))
        return

    try:
        updated = update_frozen(cfg, **{key: value})
    except ValidationError as e:
        raise click.BadParameter(str(e)) from e

    save_config(updated, data.path_manager.app_config_file)
    data.config = updated
    click.echo(f"{key} = {getattr(updated, key)}")
