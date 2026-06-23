"""Shared LiteRT helpers used across all platform backends."""

from __future__ import annotations

from typing import Any

import numpy as np


def _load_litert_interpreter(path: str, delegates: list | None = None) -> object:
    """Load and allocate a LiteRT interpreter.

    Args:
        path: Filesystem path to the ``.tflite`` model file.
        delegates: Delegate objects to pass to the interpreter. ``None`` or
            empty list uses CPU/XNNPACK (no delegates).

    Returns:
        An allocated LiteRT interpreter.

    Raises:
        RuntimeError: If a non-empty delegate list causes initialization to fail.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        from tensorflow.lite.python.interpreter import Interpreter  # noqa: PLC0415

    actual_delegates = delegates or []
    try:
        interp = Interpreter(model_path=path, experimental_delegates=actual_delegates)
    except RuntimeError as e:  # pragma: no cover
        if actual_delegates:
            msg = f"Delegate failed for {path!r}: {e}"
            raise RuntimeError(msg) from e
        raise
    interp.allocate_tensors()
    return interp


def _tflite_set_inputs(interp: Any, inputs: np.ndarray | dict[str, np.ndarray]) -> None:
    """Feed input tensors into a LiteRT interpreter.

    Args:
        interp: An allocated LiteRT interpreter.
        inputs: Single ndarray (single-input) or name→tensor dict (multi-input).

    Raises:
        KeyError: If a named input is not found in the model.
        TypeError: If a tensor dtype does not match the model's expected dtype.
    """
    input_details = interp.get_input_details()

    if isinstance(inputs, np.ndarray):
        interp.set_tensor(input_details[0]["index"], inputs)
        return

    name_to_detail = {d["name"]: d for d in input_details}
    for name, tensor in inputs.items():
        if name not in name_to_detail:
            available = list(name_to_detail.keys())
            msg = f"Input name {name!r} not found in model. Available: {available}"
            raise KeyError(msg)
        detail = name_to_detail[name]
        if tensor.dtype != detail["dtype"]:
            msg = (
                f"Input {name!r} dtype mismatch: "
                f"got {tensor.dtype}, model expects {detail['dtype']}"
            )
            raise TypeError(msg)
        interp.set_tensor(detail["index"], tensor)
