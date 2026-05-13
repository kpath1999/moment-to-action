"""Unit tests for moment_to_action.utils.video."""

from __future__ import annotations

import numpy as np
import pytest

from moment_to_action.utils.video import sample_frames, to_pil_rgb


@pytest.mark.unit
class TestToPilRgb:
    """Tests for to_pil_rgb."""

    def test_converts_bgr_to_rgb_mode(self) -> None:
        """Output PIL image has mode 'RGB'."""
        bgr = np.zeros((10, 20, 3), dtype=np.uint8)
        img = to_pil_rgb(bgr)
        assert img.mode == "RGB"

    def test_output_size_matches_input_hw(self) -> None:
        """PIL size (width, height) matches input (W, H)."""
        bgr = np.zeros((100, 200, 3), dtype=np.uint8)
        img = to_pil_rgb(bgr)
        assert img.size == (200, 100)

    def test_blue_bgr_becomes_blue_rgb(self) -> None:
        """A pure-blue BGR pixel is converted to pure-blue RGB pixel."""
        bgr = np.zeros((1, 1, 3), dtype=np.uint8)
        bgr[0, 0, 0] = 255  # Blue channel in BGR
        img = to_pil_rgb(bgr)
        r, g, b = img.getpixel((0, 0))  # type: ignore[misc]
        assert r == 0
        assert g == 0
        assert b == 255

    def test_red_bgr_becomes_red_rgb(self) -> None:
        """A pure-red BGR pixel (R channel = index 2) is correctly converted."""
        bgr = np.zeros((1, 1, 3), dtype=np.uint8)
        bgr[0, 0, 2] = 255  # Red channel in BGR
        img = to_pil_rgb(bgr)
        r, g, b = img.getpixel((0, 0))  # type: ignore[misc]
        assert r == 255
        assert g == 0
        assert b == 0

    def test_black_frame_stays_black(self) -> None:
        """All-zero BGR frame produces an all-black RGB image."""
        bgr = np.zeros((5, 5, 3), dtype=np.uint8)
        img = to_pil_rgb(bgr)
        pixels = list(img.getdata())
        assert all(p == (0, 0, 0) for p in pixels)

    def test_larger_frame(self) -> None:
        """Works for a typical video-frame size (480x640)."""
        rng = np.random.default_rng(42)
        bgr = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        img = to_pil_rgb(bgr)
        assert img.size == (640, 480)
        assert img.mode == "RGB"


@pytest.mark.unit
class TestSampleFrames:
    """Tests for sample_frames."""

    def test_returns_same_list_when_under_limit(self) -> None:
        """Returns the exact same list object when len ≤ max_images."""
        frames = [np.zeros((10, 10)) for _ in range(3)]
        result = sample_frames(frames, max_images=8)
        assert result is frames

    def test_returns_same_list_when_equal_to_limit(self) -> None:
        """Returns the exact same list object when len == max_images."""
        frames = [np.zeros((10, 10)) for _ in range(4)]
        result = sample_frames(frames, max_images=4)
        assert result is frames

    def test_samples_down_to_max_images(self) -> None:
        """Returned list has exactly max_images elements when len > max_images."""
        frames = [np.zeros((1, 1)) for _ in range(20)]
        result = sample_frames(frames, max_images=4)
        assert len(result) == 4

    def test_includes_first_and_last_frame(self) -> None:
        """Sampled result always includes the first and last frame."""
        frames = [np.full((1, 1), i, dtype=np.int32) for i in range(20)]
        result = sample_frames(frames, max_images=4)
        np.testing.assert_array_equal(result[0], frames[0])
        np.testing.assert_array_equal(result[-1], frames[-1])

    def test_single_frame_list(self) -> None:
        """A single-frame list with max_images=1 returns that frame."""
        frames = [np.zeros((10, 10))]
        result = sample_frames(frames, max_images=1)
        assert len(result) == 1
        assert result is frames

    def test_sampling_is_uniform(self) -> None:
        """Indices are uniformly spaced across the frame range."""
        # 10 frames, sample 3 → indices should be 0, 4 (round(4.5)), 9
        # step = (10-1)/(3-1) = 4.5
        # i=0 → 0, i=1 → round(4.5)=4 or 5, i=2 → round(9.0)=9
        frames = [np.full((1, 1), i, dtype=np.int32) for i in range(10)]
        result = sample_frames(frames, max_images=3)
        assert len(result) == 3
        np.testing.assert_array_equal(result[0], frames[0])
        np.testing.assert_array_equal(result[-1], frames[-1])
        # Middle frame must be one of the middle indices
        middle_val = int(result[1][0, 0])
        assert 1 <= middle_val <= 8

    def test_max_images_one_returns_first_frame(self) -> None:
        """When max_images=1, only the first frame is returned."""
        frames = [np.full((1, 1), i, dtype=np.int32) for i in range(10)]
        result = sample_frames(frames, max_images=1)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], frames[0])

    def test_preserves_frame_order(self) -> None:
        """Sampled frames maintain temporal order."""
        frames = [np.full((1, 1), i, dtype=np.int32) for i in range(100)]
        result = sample_frames(frames, max_images=10)
        values = [int(f[0, 0]) for f in result]
        assert values == sorted(values)
