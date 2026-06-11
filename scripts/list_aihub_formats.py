"""Print valid precision/runtime combinations for a qai_hub_models model."""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
import sys


def _extract_supported_precision_runtimes(main_src: str) -> dict[str, list[str]]:
    """Parse supported_precision_runtimes dict literal from main() source.

    Args:
        main_src: Source code of the main() function.

    Returns:
        Dict mapping precision name → list of runtime names.
    """
    result: dict[str, list[str]] = {}
    current_precision: str | None = None

    for line in main_src.splitlines():
        prec_match = re.search(r"Precision\.(\w+)\s*:", line)
        if prec_match:
            current_precision = prec_match.group(1)
            result[current_precision] = []
        elif current_precision:
            rt_match = re.search(r"TargetRuntime\.(\w+)", line)
            if rt_match:
                result[current_precision].append(rt_match.group(1))

    return result


def main() -> None:
    """Entry point.

    Args: none (reads from sys.argv).

    Returns: None.
    """
    parser = argparse.ArgumentParser(
        description="List valid precision/runtime combinations for a qai_hub_models model."
    )
    parser.add_argument(
        "model_id",
        help="qai_hub_models model ID (e.g. rtmdet, rf_detr, yolov8_det)",
    )
    args = parser.parse_args()

    module_path = f"qai_hub_models.models.{args.model_id}.export"
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        print(f"Error: model '{args.model_id}' not found in qai_hub_models.", file=sys.stderr)
        sys.exit(1)

    main_fn = getattr(mod, "main", None)
    if main_fn is None:
        print(f"Error: no main() in {module_path}.", file=sys.stderr)
        sys.exit(1)

    combos = _extract_supported_precision_runtimes(inspect.getsource(main_fn))
    if not combos:
        print(f"No supported_precision_runtimes found in {module_path}.main().", file=sys.stderr)
        sys.exit(1)

    print(f"Supported precision/runtime combinations for {args.model_id}:")
    for precision, runtimes in combos.items():
        for runtime in runtimes:
            print(f"  {precision} / {runtime}")


if __name__ == "__main__":
    main()
