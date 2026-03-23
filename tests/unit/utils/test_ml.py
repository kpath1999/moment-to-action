"""Tests for moment_to_action.utils.ml utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from moment_to_action.utils.ml import cosine_similarity, softmax


@pytest.mark.unit
class TestCosineSimilarity:
    """Test cosine_similarity function."""

    def test_parallel_vectors(self) -> None:
        """Parallel vectors should have cosine similarity of 1.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([2.0, 0.0, 0.0])
        result = cosine_similarity(a, b)
        assert np.isclose(result, 1.0)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors should have cosine similarity of 0.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result = cosine_similarity(a, b)
        assert np.isclose(result, 0.0, atol=1e-6)

    def test_opposite_vectors(self) -> None:
        """Opposite vectors should have cosine similarity of -1.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-2.0, 0.0, 0.0])
        result = cosine_similarity(a, b)
        assert np.isclose(result, -1.0)

    def test_zero_norm_edge_case(self) -> None:
        """Zero-norm vectors should be handled gracefully (not cause division by zero)."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        result = cosine_similarity(a, b)
        # Should not raise an exception; result may be small but valid
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_both_zero_vectors(self) -> None:
        """Both zero vectors should return a finite result."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        result = cosine_similarity(a, b)
        assert isinstance(result, float)
        assert np.isfinite(result)


@pytest.mark.unit
class TestSoftmax:
    """Test softmax function."""

    def test_uniform_input(self) -> None:
        """Uniform input should have equal softmax probabilities."""
        x = np.array([1.0, 1.0, 1.0, 1.0])
        result = softmax(x)
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        assert np.allclose(result, expected)

    def test_peaked_input(self) -> None:
        """Peaked input should concentrate probability on the maximum."""
        x = np.array([10.0, 1.0, 1.0])
        result = softmax(x)
        # First element should have much higher probability
        assert result[0] > 0.9
        assert result[1] < 0.1
        assert result[2] < 0.1

    def test_numerical_stability_large_values(self) -> None:
        """Softmax should be numerically stable with large input values."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        # Should not produce NaN or Inf
        assert np.all(np.isfinite(result))
        # Should still sum to 1
        assert np.isclose(result.sum(), 1.0)
        # Third element should have highest probability
        assert result[2] > result[1] > result[0]

    def test_softmax_sums_to_one(self) -> None:
        """Softmax output should always sum to 1."""
        x = np.array([0.5, -1.2, 3.1, -0.8])
        result = softmax(x)
        assert np.isclose(result.sum(), 1.0)

    def test_softmax_preserves_order(self) -> None:
        """Softmax should preserve the order of input values."""
        x = np.array([1.0, 3.0, 2.0])
        result = softmax(x)
        # Higher input values should have higher softmax output
        assert result[1] > result[2] > result[0]
