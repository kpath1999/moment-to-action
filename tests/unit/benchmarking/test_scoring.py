"""Unit tests for benchmarking._scoring."""

from __future__ import annotations

import numpy as np
import pytest

from moment_to_action.benchmarking import ap50, detect_yn, recall


@pytest.mark.unit
class TestRecall:
    """Tests for recall()."""

    def test_empty_keywords_is_perfect_recall(self) -> None:
        """No keywords required means recall is always 1.0."""
        assert recall("anything", []) == 1.0

    def test_all_keywords_found(self) -> None:
        """All keywords present gives recall 1.0."""
        assert recall("Yes, this is a fight.", ["yes", "fight"]) == 1.0

    def test_no_keywords_found(self) -> None:
        """No keywords present gives recall 0.0."""
        assert recall("Nothing relevant here.", ["yes", "fight"]) == 0.0

    def test_partial_keywords_found(self) -> None:
        """Half the keywords present gives recall 0.5."""
        assert recall("Yes, calm scene.", ["yes", "fight"]) == pytest.approx(0.5)

    def test_case_insensitive_match(self) -> None:
        """Keyword matching ignores case."""
        assert recall("YES definitely", ["yes"]) == 1.0


@pytest.mark.unit
class TestDetectYn:
    """Tests for detect_yn()."""

    def test_leading_yes(self) -> None:
        """A response starting with 'yes' is detected."""
        assert detect_yn("Yes, because of reasons.") == "yes"

    def test_leading_no_with_punctuation(self) -> None:
        """A response starting with 'No.' strips punctuation."""
        assert detect_yn("No.") == "no"

    def test_answer_label_pattern(self) -> None:
        """An 'Answer: YES' pattern is detected even without a leading word."""
        assert detect_yn("After analysis, Answer: YES") == "yes"

    def test_answer_label_no(self) -> None:
        """An 'Answer: No' pattern is detected."""
        assert detect_yn("Reasoning here. Answer: No") == "no"

    def test_no_decision_yet(self) -> None:
        """Text with no yes/no signal returns None."""
        assert detect_yn("I am still thinking") is None

    def test_empty_text(self) -> None:
        """Empty text returns None."""
        assert detect_yn("") is None


@pytest.mark.unit
class TestAp50:
    """Tests for ap50()."""

    def test_empty_gt_and_pred_is_perfect(self) -> None:
        """No ground truth and no predictions is a perfect score."""
        empty = np.zeros((0, 4), dtype=np.float32)
        assert ap50(empty, np.zeros(0), empty) == 1.0

    def test_empty_gt_nonempty_pred_is_zero(self) -> None:
        """Predictions with no ground truth boxes score zero."""
        gt = np.zeros((0, 4), dtype=np.float32)
        pred = np.array([[0, 0, 10, 10]], dtype=np.float32)
        assert ap50(pred, np.array([0.9]), gt) == 0.0

    def test_nonempty_gt_empty_pred_is_zero(self) -> None:
        """Ground truth with no predictions scores zero."""
        gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
        empty = np.zeros((0, 4), dtype=np.float32)
        assert ap50(empty, np.zeros(0), gt) == 0.0

    def test_perfect_match_scores_one(self) -> None:
        """A prediction exactly matching ground truth scores 1.0."""
        gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
        pred = np.array([[0, 0, 10, 10]], dtype=np.float32)
        assert ap50(pred, np.array([0.99]), gt) == pytest.approx(1.0)

    def test_poor_match_scores_low(self) -> None:
        """A prediction far from ground truth scores near zero."""
        gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
        pred = np.array([[100, 100, 110, 110]], dtype=np.float32)
        assert ap50(pred, np.array([0.9]), gt) == pytest.approx(0.0)
