"""Download a model command."""

from __future__ import annotations

import rich_click as click

from moment_to_action.models import DEFAULT_VARIANT_KEY, ModelID, ModelManager
from moment_to_action.utils.cli import GlobalData, pass_global_data


@click.command()
@click.argument("model_id", type=click.Choice([m.value for m in ModelID], case_sensitive=False))
@click.option(
    "--variant",
    default=DEFAULT_VARIANT_KEY,
    show_default=True,
    help="Variant key to download.",
)
@pass_global_data
def download(data: GlobalData, model_id: str, variant: str) -> None:
    r"""Download a model to the local cache.

    Fetches the specified variant from its registered source and stores it in
    the application model cache. If already cached, prints the existing path
    without re-downloading.

    \b
    Examples:
      m2a model download yolo_v8
      m2a model download yolo_v8 --variant qcs6490
    """
    mid = ModelID(model_id)
    path = ModelManager(data.path_manager).get_path(mid, variant)
    click.echo(f"Downloaded: {path}")
