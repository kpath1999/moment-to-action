"""Remove cached model command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import rich_click as click

from moment_to_action.models import DEFAULT_VARIANT_KEY, MODEL_REGISTRY, ModelID, ModelManager
from moment_to_action.models._sources import VendoredSource
from moment_to_action.utils.cli import GlobalData, pass_global_data

if TYPE_CHECKING:
    from moment_to_action.models._model_info import ModelInfo


def _remove_all(
    model_mgr: ModelManager,
    path_manager: object,
    model_registry: dict[ModelID, ModelInfo],
    *,
    skip_confirm: bool,
) -> None:
    """Remove all non-vendored cached model variants.

    Args:
        model_mgr: ModelManager instance for availability checks.
        path_manager: PathManager instance used for cache removal.
        model_registry: Registry mapping ModelID to ModelInfo.
        skip_confirm: If False, prompt the user for confirmation before removing.
    """
    if not skip_confirm:
        click.confirm("Remove all non-vendored cached models?", abort=True)
    total_freed = 0
    for mid, info in model_registry.items():
        for vkey, source in info.variants.items():
            if isinstance(source, VendoredSource):
                continue
            if not model_mgr.is_available(mid, vkey):
                continue
            freed = path_manager.cache.models.remove_variant(mid.value, vkey)  # type: ignore[attr-defined]
            total_freed += freed
            click.echo(f"Removed {mid.value}/{vkey} ({freed:,} B)")
    click.echo(f"Total freed: {total_freed:,} B")


@click.command()
@click.argument(
    "model_id", type=click.Choice([m.value for m in ModelID], case_sensitive=False), required=False
)
@click.option(
    "--variant",
    default=DEFAULT_VARIANT_KEY,
    show_default=True,
    help="Variant key to remove.",
)
@click.option(
    "-a", "--all", "remove_all", is_flag=True, help="Remove all non-vendored cached models."
)
@click.option("-y", "--yes", "skip_confirm", is_flag=True, help="Skip confirmation prompt.")
@pass_global_data
def remove(
    data: GlobalData,
    model_id: str | None,
    variant: str,
    *,
    remove_all: bool,
    skip_confirm: bool,
) -> None:
    r"""Remove cached model files.

    Without ``--all``: removes the specified model variant from the cache.
    With ``--all``: removes every non-vendored cached model variant.

    Vendored models (bundled with the application) cannot be removed.

    \b
    Examples:
      m2a model remove yolo_v8 --variant qcs6490
      m2a model remove --all --yes
    """
    mgr = ModelManager(data.path_manager)

    if remove_all:
        _remove_all(mgr, data.path_manager, MODEL_REGISTRY, skip_confirm=skip_confirm)
        return

    if model_id is None:
        msg = "Provide MODEL_ID or use --all."
        raise click.UsageError(msg)

    mid = ModelID(model_id)
    source = MODEL_REGISTRY[mid].variants.get(variant)
    if source is None:
        msg = f"Variant '{variant}' not found for model '{model_id}'."
        raise click.ClickException(msg)
    if isinstance(source, VendoredSource):
        msg = f"Model '{model_id}/{variant}' is vendored and cannot be removed."
        raise click.ClickException(msg)

    if not skip_confirm:
        click.confirm(f"Remove cached model {model_id}/{variant}?", abort=True)

    freed = data.path_manager.cache.models.remove_variant(mid.value, variant)
    click.echo(f"Removed {model_id}/{variant} ({freed:,} B)")
