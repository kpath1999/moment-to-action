"""Read power command."""

from __future__ import annotations

import json

import attrs
import rich_click as click

from moment_to_action.hardware import ComputeBackend, ComputeUnit


@click.command(aliases=["rdpwr"])
@click.argument("device", type=click.Choice(ComputeUnit, case_sensitive=False))
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
@click.pass_context
def read_power(ctx: click.Context, *, device: ComputeUnit, json_output: bool) -> None:  # noqa: ARG001
    """Read power of a device."""
    backend = ComputeBackend(device)
    pwr_mon = backend.power_monitor

    sample = pwr_mon.sample(device)
    if json_output:
        output = attrs.asdict(sample)
        click.echo(json.dumps(output))
    else:
        click.echo(
            f"Device {device} is drawing {sample.power_mw} mW at {sample.utilization_pct}% usage."
        )
