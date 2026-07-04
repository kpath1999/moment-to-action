"""Unit tests for stages.vlm._encode."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from moment_to_action.stages.vlm._encode import bgr_to_b64


@pytest.mark.unit
class TestBgrToB64:
    """Tests for bgr_to_b64()."""

    def test_returns_valid_base64_string(self) -> None:
        """The output decodes as valid base64."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        result = bgr_to_b64(frame)
        assert isinstance(result, str)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_decoded_bytes_are_jpeg(self) -> None:
        """The decoded bytes start with the JPEG magic number."""
        frame = np.full((20, 20, 3), 128, dtype=np.uint8)
        result = bgr_to_b64(frame)
        decoded = base64.b64decode(result)
        assert decoded[:2] == b"\xff\xd8"  # JPEG SOI marker

    def test_quality_parameter_changes_output(self) -> None:
        """Different quality settings produce different-sized output."""
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        low = base64.b64decode(bgr_to_b64(frame, quality=10))
        high = base64.b64decode(bgr_to_b64(frame, quality=95))
        assert len(low) != len(high)
