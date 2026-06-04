"""List models command."""

from __future__ import annotations

import json

import rich_click as click
from rich.console import Console
from rich.table import Table

from moment_to_action.models import MODEL_REGISTRY, ModelManager, VendoredSource
from moment_to_action.utils.cli import GlobalData, pass_global_data


@click.command()
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Output as JSON instead of a rich table.",
)
@pass_global_data
def list(data: GlobalData, *, as_json: bool) -> None:  # noqa: A001
    r"""List all models in the registry with their download status.

    Shows each model variant's format, availability (vendored / cached /
    not downloaded), size on disk, and file path.

    \b
    Examples:
      m2a model list
      m2a model list --json
    """
    mgr = ModelManager(data.path_manager)
    statuses = mgr.list_models()

    rows = []
    for model_status in statuses:
        for variant_status in model_status.variants:
            source = MODEL_REGISTRY[variant_status.model_id].variants[variant_status.variant]
            if isinstance(source, VendoredSource):
                status_str = "vendored"
            elif variant_status.available:
                status_str = "cached"
            else:
                status_str = "not downloaded"

            rows.append(
                {
                    "model_id": variant_status.model_id.value,
                    "variant": variant_status.variant,
                    "format": source.format.name,
                    "status": status_str,
                    "size_bytes": variant_status.size_bytes,
                    "path": str(variant_status.path) if variant_status.path else None,
                }
            )

    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return

    table = Table(title="Models")
    table.add_column("Model ID")
    table.add_column("Variant")
    table.add_column("Format")
    table.add_column("Status")
    table.add_column("Size")
    table.add_column("Path")

    for row in rows:
        size_bytes = row["size_bytes"]
        size_str = f"{size_bytes:,} B" if size_bytes is not None else "[dim]—[/dim]"
        path_val = row["path"]
        path_str = str(path_val) if path_val is not None else "[dim]—[/dim]"
        table.add_row(
            str(row["model_id"]),
            str(row["variant"]),
            str(row["format"]),
            str(row["status"]),
            size_str,
            path_str,
        )

    Console().print(table)
