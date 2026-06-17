"""Inspect a model command."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import rich_click as click
from rich.console import Console
from rich.table import Table

from moment_to_action.models import DEFAULT_VARIANT_KEY, MODEL_REGISTRY, ModelID, ModelManager
from moment_to_action.utils.cli import GlobalData, pass_global_data
from moment_to_action.utils.files import disk_size

if TYPE_CHECKING:
    from pathlib import Path


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: Path to the file.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


@click.command()
@click.argument("model_id", type=click.Choice([m.value for m in ModelID], case_sensitive=False))
@click.option(
    "--variant",
    default=DEFAULT_VARIANT_KEY,
    show_default=True,
    help="Variant key to inspect.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Output as JSON instead of a rich table.",
)
@pass_global_data
def inspect(data: GlobalData, model_id: str, variant: str, *, as_json: bool) -> None:
    r"""Print metadata for a model variant.

    Always shows registry metadata: source type, format, and all available
    variant keys.  When the variant is already cached, also shows the file
    size, absolute path, and SHA-256 digest.

    \b
    Examples:
      m2a model inspect yolo_v8
      m2a model inspect yolo_v8 --variant qcs6490
      m2a model inspect yolo_v8 --json
    """
    mid = ModelID(model_id)
    info = MODEL_REGISTRY[mid]
    mgr = ModelManager(data.path_manager)

    source = info.variants[variant]
    available = mgr.is_available(mid, variant)

    if available:
        path = mgr.get_path(mid, variant)
        size = disk_size(path)
        path_str = str(path)
        if path.is_file():
            sha256 = _sha256_file(path)
        else:
            n_files = sum(1 for _ in path.rglob("*") if _.is_file())
            sha256 = f"directory ({n_files} files)"
    else:
        size = None
        path_str = None
        sha256 = None

    if as_json:
        output = {
            "model_id": mid.value,
            "variant": variant,
            "source_type": type(source.source).__name__,
            "format": source.source.format.name,
            "available_variants": list(info.variants),
            "cached": available,
            "size_bytes": size,
            "path": path_str,
            "sha256": sha256,
        }
        click.echo(json.dumps(output, indent=2))
        return

    console = Console()
    table = Table(title=f"Model: {mid.value} / {variant}")
    table.add_column("Field")
    table.add_column("Value")

    table.add_row("Source type", type(source.source).__name__)
    table.add_row("Format", source.source.format.name)
    table.add_row("Available variants", ", ".join(info.variants))

    if available:
        table.add_row("Size", f"{size:,} B")
        table.add_row("Path", path_str or "")
        if sha256 and sha256.startswith("directory"):
            table.add_row("SHA-256", f"[dim]{sha256}[/dim]")
        else:
            table.add_row("SHA-256", sha256 or "")
    else:
        table.add_row("Size", "[dim]not cached[/dim]")
        table.add_row("Path", "[dim]not cached[/dim]")
        table.add_row("SHA-256", "[dim]not cached[/dim]")

    console.print(table)
