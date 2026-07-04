"""Shared glue for bench/ scripts: not library API, just bench-only plumbing.

Every bench script builds one :class:`BenchContext` (one :class:`MetricsCollector`
shared across every model/stage/pipeline constructed in that run), drives a
:class:`~moment_to_action.pipeline.Pipeline` per input inside
``with context.metrics.start_trace(): ...``, and writes results with
:func:`write_results`.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rich.console import Console

from moment_to_action.config import load_config
from moment_to_action.hardware import Platform
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models import ModelManager
from moment_to_action.paths import PathManager
from moment_to_action.qairt import QairtSDKManager

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from moment_to_action.config import AppConfig

console = Console()


@dataclass
class BenchContext:
    """Shared state for one bench script run: one collector for the whole run."""

    path_manager: PathManager
    config: AppConfig
    platform: Platform
    metrics: MetricsCollector
    manager: ModelManager


def configure_qairt(path_manager: PathManager, config: AppConfig) -> None:
    """Set up the QAIRT SDK environment (QAIRT_SDK_ROOT etc.) for DLC loading.

    Mirrors the ``m2a`` CLI root callback: without this, ``load_model_dlc``
    raises "QAIRT SDK is not available" even when the SDK is installed, because
    the environment variables are never exported into this process. A no-op
    (with a warning) if the SDK path is not configured.

    Args:
        path_manager: PathManager used to resolve QAIRT SDK cache locations.
        config: Application config holding the configured QAIRT SDK path.
    """
    if config.qairt_sdk_path is None:
        console.print(
            "  [yellow]QAIRT SDK path not configured — DLC backends may be unavailable.[/yellow]"
        )
        return
    try:
        QairtSDKManager.from_app_config(config, path_manager).configure_env()
    except RuntimeError as exc:
        console.print(f"  [yellow]QAIRT env setup failed: {exc}[/yellow]")


def build_context(*, qairt: bool = False, show_progress: bool = True) -> BenchContext:
    """Build the shared context for one bench script run.

    Args:
        qairt: Whether to configure the QAIRT SDK environment (only needed for
            detector benchmarks that may load DLC/NPU backends).
        show_progress: Forwarded to :class:`~moment_to_action.models.ModelManager`
            for download progress bars.

    Returns:
        A :class:`BenchContext` with one :class:`MetricsCollector` shared by
        every model/stage/pipeline constructed for this run.
    """
    path_manager = PathManager()
    config = load_config(path_manager.app_config_file)
    if qairt:
        configure_qairt(path_manager, config)
    platform = Platform(config)
    metrics = MetricsCollector(platform)
    manager = ModelManager(path_manager, show_progress=show_progress, metrics=metrics)
    return BenchContext(
        path_manager=path_manager,
        config=config,
        platform=platform,
        metrics=metrics,
        manager=manager,
    )


@contextlib.contextmanager
def silence_native_output() -> Iterator[None]:
    """Redirect OS-level stdout+stderr to /dev/null for the duration of the block.

    The QAIRT runtime emits C++ logger chatter (e.g. "Profile Logger with name =
    defaultKey doesn't exist!") straight to file descriptors 1/2, bypassing
    Python's logging and corrupting the rich progress bar. Wrapping native calls
    (load/run/unload) in this redirects those fds to /dev/null and restores them
    afterwards, so only Python-level output reaches the terminal.

    Yields:
        None.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(devnull)
        os.close(saved[0])
        os.close(saved[1])


def merge_by_key(
    new_entries: list[dict],
    existing_entries: list[dict],
    key_fn: Callable[[dict], object],
) -> list[dict]:
    """Merge *new_entries* into *existing_entries*, keyed by *key_fn*.

    Entries in *existing_entries* whose key also appears in *new_entries* are
    replaced; all other existing entries are kept unchanged. New entries are
    appended after the kept existing ones.

    Args:
        new_entries: Freshly produced entries from this run.
        existing_entries: Entries loaded from a previous results file.
        key_fn: Function extracting the dedup key from an entry (e.g.
            ``lambda e: e["model"]`` or ``lambda e: (e["model"], e["detector"])``).

    Returns:
        Merged list of entries.
    """
    new_keys = {key_fn(e) for e in new_entries}
    kept = [e for e in existing_entries if key_fn(e) not in new_keys]
    return kept + new_entries


def write_results(
    entries: list[dict],
    output_path: Path,
    *,
    script: str,
    entries_key: str = "models",
    key_fn: Callable[[dict], object] | None = None,
    merge: bool = False,
) -> None:
    """Write bench result entries to a JSON file, optionally merging with the existing file.

    Args:
        entries: Freshly produced result entries (typically one per model, or
            one per model/backend for the detector benchmark).
        output_path: Destination JSON path.
        script: Script name recorded in the output (``output["script"]``).
        entries_key: Top-level key under which *entries* are stored (``"models"``
            for the model-centric benches, ``"runs"`` for the detector benchmark).
        key_fn: Dedup key function for merging (required if *merge* is True).
        merge: When True, load ``output_path`` if it exists and merge *entries*
            into its existing ``entries_key`` list via :func:`merge_by_key`
            instead of overwriting it outright.
    """
    if merge and output_path.exists() and key_fn is not None:
        existing = json.loads(output_path.read_text())
        existing_entries = existing.get(entries_key, [])
        entries = merge_by_key(entries, existing_entries, key_fn)

    output = {
        "script": script,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        entries_key: entries,
    }
    output_path.write_text(json.dumps(output, indent=2))
