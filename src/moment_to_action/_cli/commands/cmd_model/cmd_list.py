"""List models command."""

from __future__ import annotations

import rich_click as click
from rich.console import Console
from rich.table import Table

from moment_to_action.models import MODEL_REGISTRY, ModelManager, VendoredSource
from moment_to_action.utils.cli import GlobalData, pass_global_data


@click.command()
@pass_global_data
def list(data: GlobalData) -> None:  # noqa: A001
    """List all models in the registry with their download status.

    Shows each model variant's format, availability (vendored / cached /
    not downloaded), size on disk, and file path.
    """
    mgr = ModelManager(data.path_manager)
    statuses = mgr.list_models()

    table = Table(title="Models")
    table.add_column("Model ID")
    table.add_column("Variant")
    table.add_column("Format")
    table.add_column("Status")
    table.add_column("Size")
    table.add_column("Path")

    for model_status in statuses:
        for variant_status in model_status.variants:
            source = MODEL_REGISTRY[variant_status.model_id].variants[variant_status.variant]
            if isinstance(source, VendoredSource):
                status_str = "vendored"
            elif variant_status.available:
                status_str = "cached"
            else:
                status_str = "not downloaded"

            size_str = (
                f"{variant_status.size_bytes:,} B"
                if variant_status.size_bytes is not None
                else "[dim]—[/dim]"
            )
            path_str = str(variant_status.path) if variant_status.path else "[dim]—[/dim]"

            table.add_row(
                variant_status.model_id.value,
                variant_status.variant,
                source.format.name,
                status_str,
                size_str,
                path_str,
            )

    Console().print(table)
