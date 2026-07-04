"""Metrics-report helpers for benchmark scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from moment_to_action.metrics import SpanType

if TYPE_CHECKING:
    from collections.abc import Iterable

    from moment_to_action.metrics import Trace


def extract_load_unload_ms(traces: Iterable[Trace], name_contains: str = "") -> tuple[float, float]:
    """Extract load and unload latencies from one or more traces.

    When the traces contain spans from multiple models (e.g. a detector and an
    LLM sharing one trace), pass *name_contains* to restrict matching to spans
    whose name includes that substring (e.g. ``"LlamaGGUFModel"``).

    Args:
        traces: Traces to scan — pass ``report.traces`` for a whole
            :class:`~moment_to_action.metrics.MetricsReport`, or ``[trace]`` for
            a single trace (e.g. one bench script model run sharing a
            longer-lived collector with other runs).
        name_contains: Optional substring filter on span name. Empty string
            (the default) matches every span.

    Returns:
        Tuple of ``(load_ms, unload_ms)``. Returns ``0.0`` for any span not found.
    """
    load_ms = 0.0
    unload_ms = 0.0
    for trace in traces:
        for span in trace.spans:
            if name_contains and name_contains not in span.name:
                continue
            if span.type_ == SpanType.MODEL_LOAD:
                load_ms = span.latency_ms
            elif span.type_ == SpanType.MODEL_UNLOAD:
                unload_ms = span.latency_ms
    return load_ms, unload_ms
