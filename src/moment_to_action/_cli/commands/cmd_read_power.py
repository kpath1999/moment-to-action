"""Read power command."""

from __future__ import annotations

import json

import attrs
import rich_click as click

from moment_to_action.hardware import ComputeUnit, Platform
from moment_to_action.utils.cli import get_global_data


@click.command(aliases=["rdpwr"])
@click.argument("device", type=click.Choice(ComputeUnit, case_sensitive=False))
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
@click.pass_context
def read_power(ctx: click.Context, *, device: ComputeUnit, json_output: bool) -> None:
    """Read power of a device."""
    data = get_global_data(ctx)
    platform = Platform(data.config)
    resource_mon = platform.resource_monitor

    sample = resource_mon.sample(device)
    if json_output:
        output = attrs.asdict(sample)
        click.echo(json.dumps(output))
    else:
        click.echo(f"Device {device} is drawing {sample.power_mw} mW at {sample.usage_pct}% usage.")
