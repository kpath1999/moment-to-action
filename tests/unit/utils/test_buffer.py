"""Tests for moment_to_action.utils.buffer buffer management."""

from __future__ import annotations

import numpy as np
import pytest

from moment_to_action.utils.buffer import BufferPool, BufferSpec


@pytest.mark.unit
class TestBufferSpec:
    """Test BufferSpec data class."""

    def test_allocate_creates_correct_shape(self) -> None:
        """BufferSpec.allocate should create an array with the correct shape."""
        spec = BufferSpec(shape=(15360,), dtype=np.float32)
        buf = spec.allocate()
        assert buf.shape == (15360,)

    def test_allocate_creates_correct_dtype(self) -> None:
        """BufferSpec.allocate should create an array with the correct dtype."""
        spec = BufferSpec(shape=(15360,), dtype=np.float32)
        buf = spec.allocate()
        assert buf.dtype == np.float32

    def test_allocate_multidimensional_shape(self) -> None:
        """BufferSpec.allocate should handle multidimensional shapes."""
        spec = BufferSpec(shape=(3, 256, 256), dtype=np.float32)
        buf = spec.allocate()
        assert buf.shape == (3, 256, 256)
        assert buf.dtype == np.float32

    def test_allocate_different_dtypes(self) -> None:
        """BufferSpec.allocate should support different dtypes."""
        for dtype in [np.float32, np.float64, np.int32, np.uint8]:
            spec = BufferSpec(shape=(100,), dtype=dtype)
            buf = spec.allocate()
            assert buf.dtype == dtype

    def test_allocate_zeroed(self) -> None:
        """BufferSpec.allocate should return a zero-filled array."""
        spec = BufferSpec(shape=(10, 10), dtype=np.float32)
        buf = spec.allocate()
        assert np.all(buf == 0.0)


@pytest.mark.unit
class TestBufferPool:
    """Test BufferPool buffer management."""

    def test_register_and_get(self) -> None:
        """BufferPool should register and retrieve buffers."""
        pool = BufferPool()
        spec = BufferSpec(shape=(100,), dtype=np.float32)
        pool.register("test_buf", spec)
        buf = pool.get("test_buf")
        assert buf.shape == (100,)
        assert buf.dtype == np.float32

    def test_get_or_register_first_call_creates(self) -> None:
        """First call to get_or_register should create the buffer."""
        pool = BufferPool()
        spec = BufferSpec(shape=(50,), dtype=np.float32)
        buf1 = pool.get_or_register("new_buf", spec)
        assert buf1.shape == (50,)

    def test_get_or_register_second_reuses(self) -> None:
        """Second call to get_or_register should return the same buffer."""
        pool = BufferPool()
        spec = BufferSpec(shape=(50,), dtype=np.float32)
        buf1 = pool.get_or_register("reused_buf", spec)
        buf1[0] = 42.0  # Modify the buffer
        buf2 = pool.get_or_register("reused_buf", spec)
        # Should be the same buffer object (same memory location)
        assert buf2[0] == 42.0
        assert buf1 is buf2

    def test_duplicate_register_name_raises_error(self) -> None:
        """Registering the same name twice without overwrite should not raise."""
        pool = BufferPool()
        spec1 = BufferSpec(shape=(100,), dtype=np.float32)
        spec2 = BufferSpec(shape=(200,), dtype=np.float32)
        pool.register("buf", spec1)
        # Second register without overwrite should be silently ignored
        pool.register("buf", spec2, overwrite=False)
        buf = pool.get("buf")
        assert buf.shape == (100,)  # Should still be the original spec

    def test_duplicate_register_name_with_overwrite(self) -> None:
        """Registering with overwrite=True should replace the buffer."""
        pool = BufferPool()
        spec1 = BufferSpec(shape=(100,), dtype=np.float32)
        spec2 = BufferSpec(shape=(200,), dtype=np.float32)
        pool.register("buf", spec1)
        pool.register("buf", spec2, overwrite=True)
        buf = pool.get("buf")
        assert buf.shape == (200,)  # Should be the new spec

    def test_get_nonexistent_buffer_raises_keyerror(self) -> None:
        """Getting a non-existent buffer should raise KeyError."""
        pool = BufferPool()
        with pytest.raises(KeyError, match="Buffer 'nonexistent' not registered"):
            pool.get("nonexistent")

    def test_total_bytes_calculation_single_buffer(self) -> None:
        """total_bytes should correctly calculate bytes for a single buffer."""
        pool = BufferPool()
        spec = BufferSpec(shape=(100,), dtype=np.float32)  # 100 * 4 = 400 bytes
        pool.register("buf", spec)
        assert pool.total_bytes == 400

    def test_total_bytes_calculation_multiple_buffers(self) -> None:
        """total_bytes should correctly sum bytes across all buffers."""
        pool = BufferPool()
        spec1 = BufferSpec(shape=(100,), dtype=np.float32)  # 400 bytes
        spec2 = BufferSpec(shape=(50,), dtype=np.float64)  # 400 bytes
        pool.register("buf1", spec1)
        pool.register("buf2", spec2)
        assert pool.total_bytes == 800

    def test_total_bytes_empty_pool(self) -> None:
        """total_bytes on an empty pool should return 0."""
        pool = BufferPool()
        assert pool.total_bytes == 0

    def test_multiple_buffers_are_independent(self) -> None:
        """Buffers in the pool should be independent."""
        pool = BufferPool()
        spec1 = BufferSpec(shape=(10,), dtype=np.float32)
        spec2 = BufferSpec(shape=(10,), dtype=np.float32)
        pool.register("buf1", spec1)
        pool.register("buf2", spec2)
        buf1 = pool.get("buf1")
        buf2 = pool.get("buf2")
        buf1[0] = 99.0
        assert buf2[0] == 0.0  # buf2 should not be affected
