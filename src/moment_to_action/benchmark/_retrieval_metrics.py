from __future__ import annotations

from collections.abc import Mapping, Sequence  # noqa: TC003

import attrs
import numpy as np

_TOP_1 = 1
_TOP_5 = 5
_TOP_10 = 10
_MIN_ITEMS_FOR_RANK = 2


@attrs.frozen
class RetrievalMetrics:
    """Retrieval quality summary against oracle ranking signals."""

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    kendall_tau: float
    mean_rank_delta: float


def compute_retrieval_metrics(
    predictions: Mapping[str, Sequence[float]],
    ground_truth: Mapping[str, Sequence[float]],
) -> RetrievalMetrics:
    """Compute retrieval metrics for image-text similarity rankings.

    Args:
        predictions: Image name -> predicted similarity scores.
        ground_truth: Image name -> oracle similarity scores.

    Returns:
        RetrievalMetrics summary.
    """
    common_names = sorted(set(predictions) & set(ground_truth))
    if not common_names:
        return RetrievalMetrics(
            recall_at_1=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            kendall_tau=0.0,
            mean_rank_delta=0.0,
        )

    hit_1 = 0
    hit_5 = 0
    hit_10 = 0
    rank_deltas: list[float] = []
    taus: list[float] = []

    for image_name in common_names:
        pred_scores = np.asarray(predictions[image_name], dtype=np.float32)
        gt_scores = np.asarray(ground_truth[image_name], dtype=np.float32)

        if pred_scores.ndim != 1 or gt_scores.ndim != 1:
            continue
        if pred_scores.size == 0 or gt_scores.size == 0:
            continue
        if pred_scores.size != gt_scores.size:
            continue

        pred_rank = _argsort_desc(pred_scores)
        gt_rank = _argsort_desc(gt_scores)
        oracle_top = int(gt_rank[0])
        pred_position = int(np.where(pred_rank == oracle_top)[0][0]) + 1

        if pred_position <= _TOP_1:
            hit_1 += 1
        if pred_position <= _TOP_5:
            hit_5 += 1
        if pred_position <= _TOP_10:
            hit_10 += 1

        rank_deltas.append(float(abs(pred_position - _TOP_1)))
        taus.append(_kendall_tau(pred_scores, gt_scores))

    count = len(rank_deltas)
    if count == 0:
        return RetrievalMetrics(
            recall_at_1=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            kendall_tau=0.0,
            mean_rank_delta=0.0,
        )

    return RetrievalMetrics(
        recall_at_1=hit_1 / count,
        recall_at_5=hit_5 / count,
        recall_at_10=hit_10 / count,
        kendall_tau=float(np.mean(np.asarray(taus, dtype=np.float32))),
        mean_rank_delta=float(np.mean(np.asarray(rank_deltas, dtype=np.float32))),
    )


def _argsort_desc(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, kind="stable")


def _kendall_tau(pred_scores: np.ndarray, gt_scores: np.ndarray) -> float:
    n_items = pred_scores.size
    if n_items < _MIN_ITEMS_FOR_RANK:
        return 0.0

    concordant = 0
    discordant = 0

    for left in range(n_items - 1):
        for right in range(left + 1, n_items):
            pred_diff = pred_scores[left] - pred_scores[right]
            gt_diff = gt_scores[left] - gt_scores[right]
            if pred_diff == 0.0 or gt_diff == 0.0:
                continue
            if pred_diff * gt_diff > 0.0:
                concordant += 1
            else:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total
