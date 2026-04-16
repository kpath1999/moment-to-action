"""YAMNet audio preprocessor.

Converts raw audio input (file path or mic waveform) into an AudioMessage
formatted exactly as YAMNet expects:

  * Mono (downmix if stereo)
  * float32 in [-1.0, 1.0]
  * 16 000 Hz sample rate
  * Minimum 0.96 s (one YAMNet frame) — shorter clips are zero-padded

The preprocessor does NOT chunk the waveform into fixed-length frames —
YAMNet accepts a variable-length 1-D waveform and returns a
per-frame score matrix itself.  Chunking would only be needed for
streaming, which can be layered on top later.

Input:  AudioInput
Output: AudioMessage
"""

from __future__ import annotations

import logging
from math import gcd
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from moment_to_action.hardware import ComputeUnit
from moment_to_action.messages.audio import AudioMessage
from moment_to_action.preprocessors._audio_input import AudioInput
from moment_to_action.preprocessors._preprocess import BasePreprocessor
from moment_to_action.utils.buffer import BufferPool, BufferSpec

logger = logging.getLogger(__name__)

# ── YAMNet constants ─────────────────────────────────────────────────────────
_TARGET_SR = 16_000          # Hz
_FRAME_SAMPLES = 15_360      # 0.96 s × 16 000 Hz — minimum clip length
_BUFFER_SECONDS = 5          # default pre-allocated buffer size (seconds)
_BUFFER_SAMPLES = _TARGET_SR * _BUFFER_SECONDS


def _load_audio(path: Path) -> tuple[NDArray[np.float32], int]:
    """Load audio from a file using soundfile, with ffmpeg fallback for mp4/m4a.

    Returns (mono float32 waveform, sample_rate).
    """
    try:
        import soundfile as sf  # type: ignore[import]
        waveform, sr = sf.read(str(path), dtype="float32", always_2d=True)
        # soundfile returns (samples, channels) — downmix to mono
        mono = waveform.mean(axis=1)
        return mono.astype(np.float32), sr

    except Exception as sf_err:
        # soundfile can't read mp4/aac — fall back to ffmpeg via subprocess
        logger.debug("soundfile failed (%s), trying ffmpeg fallback", sf_err)
        return _load_via_ffmpeg(path)


def _load_via_ffmpeg(path: Path) -> tuple[NDArray[np.float32], int]:
    """Decode audio from any container (mp4, mkv, …) via ffmpeg subprocess.

    Outputs raw 32-bit float PCM at the file's native sample rate.
    Requires ffmpeg to be installed and on $PATH.
    """
    import subprocess

    cmd = [
        "ffmpeg",
        "-v", "quiet",
        "-i", str(path),
        "-f", "f32le",       # raw 32-bit float little-endian
        "-ac", "1",          # force mono
        "-ar", "0",          # keep native sample rate (we resample later)
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on {path}:\n{result.stderr.decode()}"
        )

    waveform = np.frombuffer(result.stdout, dtype=np.float32)

    # Extract native sample rate via ffprobe
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
    try:
        sr = int(probe.stdout.strip())
    except ValueError:
        sr = _TARGET_SR
        logger.warning("Could not determine sample rate from %s, assuming %d Hz", path, sr)

    return waveform, sr


def _resample(waveform: NDArray[np.float32], src_sr: int, dst_sr: int) -> NDArray[np.float32]:
    """Resample waveform from src_sr to dst_sr using scipy.signal.resample_poly.

    resample_poly uses a polyphase filter — more accurate than naive
    resampling and avoids the memory spike of scipy.signal.resample.
    Computes the up/down ratio from the GCD to keep filter taps small.
    """
    if src_sr == dst_sr:
        return waveform
    g = gcd(src_sr, dst_sr)
    up, down = dst_sr // g, src_sr // g
    from scipy.signal import resample_poly  # type: ignore[import]
    resampled = resample_poly(waveform, up, down)
    return resampled.astype(np.float32)


def _normalise(waveform: NDArray[np.float32]) -> NDArray[np.float32]:
    """Peak-normalise to [-1.0, 1.0].  No-op if waveform is already silent."""
    peak = np.abs(waveform).max()
    if peak > 1e-6:
        return waveform / peak
    return waveform


class YAMNetPreprocessor(BasePreprocessor[AudioInput, AudioMessage]):
    """Prepare raw audio for YAMNet TFLite inference.

    Handles both offline files and live mic waveforms via AudioInput.
    The output AudioMessage waveform is ready to be set directly on the
    TFLite interpreter input tensor.

    Parameters
    ----------
    compute_unit:
        CPU or DSP routing for dispatch.  Defaults to CPU.
    buffer_seconds:
        Pre-allocated waveform buffer size in seconds.
        Clips longer than this are NOT truncated — the buffer is only
        used for clips that fit; longer clips allocate fresh arrays.
        Defaults to 5 s.
    normalise:
        If True (default), peak-normalise the waveform to [-1, 1].
        Disable if your capture stage already guarantees this.

    Examples
    --------
    From a file (testing)::

        pre = YAMNetPreprocessor()
        msg = pre.process(AudioInput(path="clip.mp4"))
        # msg.waveform is ready for YAMNetStage

    From a mic waveform (deployment)::

        pre = YAMNetPreprocessor()
        msg = pre.process(AudioInput(waveform=pcm_array, sample_rate=44100))
    """

    def __init__(
        self,
        compute_unit: ComputeUnit = ComputeUnit.CPU,
        buffer_seconds: int = _BUFFER_SECONDS,
        normalise: bool = True,
    ) -> None:
        self._buffer_seconds = buffer_seconds
        self._normalise = normalise
        super().__init__(compute_unit)

    # ------------------------------------------------------------------
    # BasePreprocessor interface
    # ------------------------------------------------------------------

    def _allocate_buffers(self) -> None:
        """Pre-allocate a waveform buffer for typical-length clips."""
        self._buffers.register(
            "waveform",
            BufferSpec((_TARGET_SR * self._buffer_seconds,), np.float32),
        )

    def _validate(self, data: AudioInput) -> None:
        if data.path is not None and not data.path.exists():
            raise ValueError(f"Audio file not found: {data.path}")
        if data.waveform is not None and data.waveform.ndim != 1:
            raise ValueError(
                f"Waveform must be 1-D mono, got shape {data.waveform.shape}. "
                "Downmix to mono before passing to YAMNetPreprocessor."
            )
        if data.sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {data.sample_rate}")

    def _process(self, data: AudioInput) -> AudioMessage:
        # 1. Load from file or use provided waveform
        if data.path is not None:
            waveform, src_sr = _load_audio(data.path)
        else:
            waveform = data.waveform.astype(np.float32)
            src_sr = data.sample_rate

        # 2. Resample to 16 kHz
        waveform = self._dispatch(_resample, waveform, src_sr, _TARGET_SR)

        # 3. Normalise
        if self._normalise:
            waveform = self._dispatch(_normalise, waveform)

        # 4. Pad to at least one YAMNet frame (0.96 s = 15 360 samples)
        if len(waveform) < _FRAME_SAMPLES:
            pad = _FRAME_SAMPLES - len(waveform)
            waveform = np.pad(waveform, (0, pad), mode="constant")
            logger.debug(
                "YAMNetPreprocessor: padded short clip by %d samples (%.2f s)",
                pad, pad / _TARGET_SR,
            )

        # 5. Write into pre-allocated buffer if it fits; otherwise use the array directly
        n = len(waveform)
        buf = self._buffers.get("waveform")
        if n <= len(buf):
            buf[:n] = waveform
            out = buf[:n].copy()
        else:
            # Clip longer than pre-allocated buffer — allocate fresh (uncommon)
            logger.debug(
                "YAMNetPreprocessor: clip (%d samples) exceeds buffer (%d) — fresh alloc",
                n, len(buf),
            )
            out = waveform.copy()

        logger.debug(
            "YAMNetPreprocessor: ready  samples=%d  duration=%.2fs  sr=%d",
            len(out), len(out) / _TARGET_SR, _TARGET_SR,
        )

        import time
        return AudioMessage(
            waveform=out,
            sample_rate=_TARGET_SR,
            source_path=data.source_label,
            timestamp=time.time(),
        )
