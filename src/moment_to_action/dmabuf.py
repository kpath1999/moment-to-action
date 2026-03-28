import ctypes
import mmap
import os
import struct
import numpy as np

# ── Step 1: define the ioctl for DMA-BUF heap allocation ──────────
# These constants come from the Linux kernel header:
# include/uapi/linux/dma-heap.h
# The kernel expects this exact struct when you call IOCTL_HEAP_ALLOC

# struct dma_heap_allocation_data {
#     __u64 len;          // how many bytes you want
#     __u32 fd;           // kernel fills this in with the buffer fd
#     __u32 fd_flags;     // file flags for the returned fd (use O_RDWR|O_CLOEXEC)
#     __u64 heap_flags;   // heap-specific flags (0 = default)
# };
DMA_HEAP_ALLOC_STRUCT = struct.Struct("QIIQ")   # u64, u32, u32, u64

# ioctl number for DMA_HEAP_IOCTL_ALLOC
# Linux encodes direction+size+type+number into one integer
# This value is fixed by the kernel ABI — it never changes
DMA_HEAP_IOCTL_ALLOC = 0xC0184800

O_RDWR    = 0x2
O_CLOEXEC = 0x80000

def allocate_dma_buf(heap_path: str, size: int) -> int:
    """Open the heap and allocate a buffer. Returns a file descriptor."""
    heap_fd = os.open(heap_path, O_RDWR | O_CLOEXEC)

    # Build the request struct — kernel fills in 'fd' for us
    buf = bytearray(DMA_HEAP_ALLOC_STRUCT.size)
    DMA_HEAP_ALLOC_STRUCT.pack_into(buf, 0,
        size,          # len — how much memory we want
        0,             # fd  — kernel writes the result here
        O_RDWR | O_CLOEXEC,  # fd_flags
        0,             # heap_flags — 0 means default
    )

    # Call the kernel
    fcntl = ctypes.CDLL("libc.so.6", use_errno=True).ioctl
    fcntl.restype = ctypes.c_int
    arr = (ctypes.c_char * len(buf)).from_buffer(buf)
    ret = fcntl(heap_fd, DMA_HEAP_IOCTL_ALLOC, arr)
    os.close(heap_fd)

    if ret != 0:
        raise OSError(ctypes.get_errno(), "DMA-BUF allocation failed")

    # Unpack the fd the kernel filled in
    _, buf_fd, _, _ = DMA_HEAP_ALLOC_STRUCT.unpack_from(buf)
    return buf_fd


# ── Step 2: map it and use it as numpy ────────────────────────────
SIZE = 640 * 640 * 3   # one 640x640 RGB frame (~1.2 MB)

print(f"Allocating {SIZE/1024/1024:.1f} MB DMA-BUF buffer...")
buf_fd = allocate_dma_buf("/dev/dma_heap/system", SIZE)
print(f"Got buffer fd: {buf_fd}")

# Map into Python process memory
mem = mmap.mmap(buf_fd, SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

# Wrap as numpy — zero copy, this IS the DMA-BUF memory
frame = np.frombuffer(mem, dtype=np.uint8).reshape(640, 640, 3)

# Write something
frame[:, :, 0] = 42   # set red channel to 42 everywhere
frame[0, 0, :] = [255, 128, 64]

# Read it back — proves we're reading the same memory
print(f"frame[0,0] = {frame[0,0]}  (expected [255, 128,  64])")
print(f"frame[1,1] = {frame[1,1]}  (expected [ 42,   0,   0])")
print("Success — DMA-BUF buffer works as numpy array")

# Cleanup
mem.close()
os.close(buf_fd)
print("Buffer freed")
