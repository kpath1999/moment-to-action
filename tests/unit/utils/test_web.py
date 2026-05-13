"""Unit tests for moment_to_action.utils.web."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING
from unittest import mock

import pytest

from moment_to_action.utils.web import download_file, stream_with_progress

if TYPE_CHECKING:
    from pathlib import Path


def _stream_ctx(response: mock.MagicMock) -> mock.MagicMock:
    """Wrap a response mock in a context-manager-compatible mock for httpx.stream."""
    ctx = mock.MagicMock()
    ctx.__enter__ = mock.MagicMock(return_value=response)
    ctx.__exit__ = mock.MagicMock(return_value=False)
    return ctx


@pytest.mark.unit
class TestStreamWithProgress:
    """Tests for `stream_with_progress`."""

    def test_writes_all_chunks_with_explicit_total(self) -> None:
        """When `total` is given, the progress task is registered with it."""
        response = mock.MagicMock()
        response.iter_bytes.return_value = [b"abc", b"def"]
        progress = mock.MagicMock()
        progress.add_task.return_value = 0
        buf = io.BytesIO()

        written = stream_with_progress(response, buf, progress, "x", total=6)

        assert written == 6
        assert buf.getvalue() == b"abcdef"
        progress.add_task.assert_called_once_with("x", total=6)
        assert progress.update.call_count == 2

    def test_uses_content_length_when_total_is_none(self) -> None:
        """When `total` is None, falls back to the Content-Length header."""
        response = mock.MagicMock()
        response.headers = {"Content-Length": "5"}
        response.iter_bytes.return_value = [b"hello"]
        progress = mock.MagicMock()
        progress.add_task.return_value = 0
        buf = io.BytesIO()

        written = stream_with_progress(response, buf, progress, "y", total=None)

        assert written == 5
        progress.add_task.assert_called_once_with("y", total=5)


@pytest.mark.unit
class TestDownloadFile:
    """Tests for `download_file`."""

    def test_writes_chunks_without_progress(self, tmp_path: Path) -> None:
        """show_progress=False writes chunks directly to dest_path."""
        response = mock.MagicMock()
        response.iter_bytes.return_value = [b"foo", b"bar"]
        dest = tmp_path / "out.bin"

        with mock.patch("httpx.stream", return_value=_stream_ctx(response)):
            written = download_file("http://x", dest, show_progress=False)

        assert written == 6
        assert dest.read_bytes() == b"foobar"

    def test_writes_chunks_with_progress(self, tmp_path: Path) -> None:
        """show_progress=True writes chunks via the Rich progress bar."""
        response = mock.MagicMock()
        response.iter_bytes.return_value = [b"xyz"]
        response.headers = {}
        dest = tmp_path / "out.bin"

        with (
            mock.patch("httpx.stream", return_value=_stream_ctx(response)),
            mock.patch("rich.progress.Progress.start"),
            mock.patch("rich.progress.Progress.stop"),
            mock.patch("rich.progress.Progress.add_task", return_value=0),
            mock.patch("rich.progress.Progress.update"),
        ):
            written = download_file("http://x", dest, show_progress=True, total=3)

        assert written == 3
        assert dest.read_bytes() == b"xyz"

    def test_raises_for_status(self, tmp_path: Path) -> None:
        """A non-2xx response surfaces the underlying HTTP error."""
        response = mock.MagicMock()
        response.raise_for_status.side_effect = RuntimeError("boom")
        dest = tmp_path / "out.bin"

        with mock.patch("httpx.stream", return_value=_stream_ctx(response)):
            with pytest.raises(RuntimeError, match="boom"):
                download_file("http://x", dest, show_progress=False)
