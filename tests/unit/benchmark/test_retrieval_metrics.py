"""Unit tests for retrieval ranking metric helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest

from moment_to_action.benchmark._retrieval_metrics import _kendall_tau, compute_retrieval_metrics


@pytest.mark.unit
def test_compute_retrieval_metrics_top_k_hits() -> None:
    """Perfect alignment should produce full recall and zero rank delta."""
    predictions = {
        "a.jpg": [0.9, 0.3, 0.1],
        "b.jpg": [0.1, 0.7, 0.2],
    }
    ground_truth = {
        "a.jpg": [0.8, 0.2, 0.1],
        "b.jpg": [0.2, 0.6, 0.3],
    }

    metrics = compute_retrieval_metrics(predictions=predictions, ground_truth=ground_truth)

    assert metrics.recall_at_1 == pytest.approx(1.0)
    assert metrics.recall_at_5 == pytest.approx(1.0)
    assert metrics.recall_at_10 == pytest.approx(1.0)
    assert -1.0 <= metrics.kendall_tau <= 1.0
    assert metrics.mean_rank_delta == pytest.approx(0.0)


@pytest.mark.unit
def test_compute_retrieval_metrics_handles_empty_overlap() -> None:
    """No shared keys should return zeroed retrieval metrics."""
    metrics = compute_retrieval_metrics(predictions={"a": [0.1]}, ground_truth={"b": [0.2]})
    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_5 == 0.0
    assert metrics.recall_at_10 == 0.0


@pytest.mark.unit
def test_compute_retrieval_metrics_skips_invalid_shapes_and_sizes() -> None:
    predictions: dict[str, Sequence[float]] = {
        "a": cast("Sequence[float]", np.array([[0.1, 0.2]], dtype=np.float32)),
        "b": cast("Sequence[float]", np.array([], dtype=np.float32)),
        "c": [0.2, 0.3],
    }
    metrics = compute_retrieval_metrics(
        predictions=predictions,
        ground_truth={
            "a": [0.2, 0.1],
            "b": [0.1],
            "c": [0.2],
        },
    )
    assert metrics.recall_at_1 == 0.0
    assert metrics.mean_rank_delta == 0.0


@pytest.mark.unit
def test_kendall_tau_edge_cases() -> None:
    assert _kendall_tau(np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32)) == 0.0
    assert (
        _kendall_tau(np.array([1.0, 1.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32))
        == 0.0
    )


@pytest.mark.unit
def test_kendall_tau_discordant_pairs() -> None:
    pred = np.array([0.1, 0.9, 0.2], dtype=np.float32)
    gt = np.array([0.9, 0.1, 0.2], dtype=np.float32)
    assert _kendall_tau(pred, gt) < 0.0
