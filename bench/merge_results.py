#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = []
# ///
"""Merge two or more benchmark_real JSON result files into one.

Entries are keyed by (model, detector) — later files win on conflict.

Usage:
    uv run python bench/merge_results.py FILE [FILE ...] --output OUTPUT
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Merge benchmark_real result JSON files.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Input JSON files to merge (later files win on conflict).",
    )
    parser.add_argument(
        "--output",
        default="bench/results/results_combined.json",
        help="Output path (default: bench/results/results_combined.json).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the merge script."""
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries: dict[tuple[str, str], dict] = {}
    base: dict = {}

    for p in args.files:
        data = json.loads(Path(p).read_text())
        if not base:
            base = {k: v for k, v in data.items() if k != "models"}
        for entry in data.get("models", []):
            key = (entry["model"], entry.get("detector", ""))
            entries[key] = entry

    base["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
    base["models"] = list(entries.values())

    output_path.write_text(json.dumps(base, indent=2))
    print(f"Merged {len(entries)} model entries from {len(args.files)} files → {output_path}")


if __name__ == "__main__":
    main()
