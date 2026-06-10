"""Unit tests for moment_to_action.models._sources resolvers."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from moment_to_action.models import (
    DownloadSource,
    HuggingFaceSource,
    ModelFormat,
    UltralyticsSource,
    VendoredSource,
    resolve_download_source,
    resolve_hugging_face_source,
    resolve_model_source,
    resolve_vendored_source,
)

# ---------------------------------------------------------------------------
# resolve_vendored_source
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveVendored:
    """Tests for resolve_vendored_source."""

    def test_returns_absolute_path_under_vendored(self) -> None:
        """The resolved path lives under `<models>/_vendored/<source.path>`."""
        s = VendoredSource(format=ModelFormat.ONNX, path=Path("yolo/model.onnx"))
        resolved = resolve_vendored_source(s)
        assert resolved.is_absolute()
        assert resolved.parts[-3:] == ("_vendored", "yolo", "model.onnx")


# ---------------------------------------------------------------------------
# resolve_download_source
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveDownload:
    """Tests for resolve_download_source."""

    def _source(self) -> DownloadSource:
        return DownloadSource(
            format=ModelFormat.ONNX,
            url="https://example.com/m.bin",
            filename="m.bin",
        )

    def test_returns_existing_target_without_download(self, tmp_path: Path) -> None:
        """If the target already exists, returns it without calling download_file."""
        s = self._source()
        (tmp_path / "m.bin").write_text("ok")

        with mock.patch("moment_to_action.models._sources._download.download_file") as mock_dl:
            result = resolve_download_source(s, tmp_path)

        assert result == tmp_path / "m.bin"
        mock_dl.assert_not_called()

    def test_missing_without_download_returns_none(self, tmp_path: Path) -> None:
        """Missing target + download=False yields None."""
        s = self._source()
        assert resolve_download_source(s, tmp_path, download=False) is None

    def test_missing_with_download_triggers_download(self, tmp_path: Path) -> None:
        """Missing target + download=True calls download_file and returns the file path."""
        s = self._source()

        def fake_dl(url: str, dest: Path, *, show_progress: bool) -> None:  # noqa: ARG001
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("downloaded")

        with mock.patch(
            "moment_to_action.models._sources._download.download_file",
            side_effect=fake_dl,
        ) as mock_dl:
            result = resolve_download_source(s, tmp_path, download=True, progress=False)

        assert result == tmp_path / "m.bin"
        mock_dl.assert_called_once()

    def test_download_did_not_produce_file_raises(self, tmp_path: Path) -> None:
        """When download_file returns but the file is still missing, raise RuntimeError."""
        s = self._source()
        with mock.patch(
            "moment_to_action.models._sources._download.download_file",
            return_value=None,
        ):
            with pytest.raises(RuntimeError, match="Failed to download"):
                resolve_download_source(s, tmp_path, download=True)


# ---------------------------------------------------------------------------
# resolve_hugging_face_source
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveHuggingFace:
    """Tests for resolve_hugging_face_source."""

    def _source(self) -> HuggingFaceSource:
        return HuggingFaceSource(
            format=ModelFormat.ONNX,
            hf_repo_id="org/repo",
            files=["a.bin", "b.bin"],
            revision="rev",
        )

    def test_all_files_present_returns_variant_dir(self, tmp_path: Path) -> None:
        """When every listed file is present, returns variant_dir without downloading."""
        s = self._source()
        for fn in s.files:
            (tmp_path / fn).write_text("ok")

        with mock.patch("moment_to_action.models._sources._hugging_face.download_file") as mock_dl:
            result = resolve_hugging_face_source(s, tmp_path)

        assert result == tmp_path
        mock_dl.assert_not_called()

    def test_missing_without_download_returns_none(self, tmp_path: Path) -> None:
        """Any missing file with download=False yields None."""
        s = self._source()
        assert resolve_hugging_face_source(s, tmp_path, download=False) is None

    def test_missing_with_download_invokes_downloads(self, tmp_path: Path) -> None:
        """download=True fetches each missing file via download_file."""
        s = self._source()

        def fake_dl(_url: str, dest: Path, **_kwargs: object) -> None:
            dest.write_text("got")

        meta = mock.MagicMock()
        meta.size = 42

        with (
            mock.patch(
                "moment_to_action.models._sources._hugging_face.hf_hub_url",
                return_value="https://hf.co/file",
            ),
            mock.patch(
                "moment_to_action.models._sources._hugging_face.get_hf_file_metadata",
                return_value=meta,
            ),
            mock.patch(
                "moment_to_action.models._sources._hugging_face.download_file",
                side_effect=fake_dl,
            ) as mock_dl,
        ):
            result = resolve_hugging_face_source(s, tmp_path, download=True, progress=False)

        assert result == tmp_path
        assert mock_dl.call_count == len(s.files)
        for fn in s.files:
            assert (tmp_path / fn).exists()

    def test_download_missing_after_attempt_raises(self, tmp_path: Path) -> None:
        """If a file is still missing after download, raise RuntimeError."""
        s = self._source()
        meta = mock.MagicMock()
        meta.size = 1

        with (
            mock.patch(
                "moment_to_action.models._sources._hugging_face.hf_hub_url",
                return_value="https://hf.co/file",
            ),
            mock.patch(
                "moment_to_action.models._sources._hugging_face.get_hf_file_metadata",
                return_value=meta,
            ),
            # Stub download_file but write nothing — file remains missing
            mock.patch("moment_to_action.models._sources._hugging_face.download_file"),
        ):
            with pytest.raises(RuntimeError, match="Failed to download"):
                resolve_hugging_face_source(s, tmp_path, download=True)

    def test_hf_subdir_prefixes_download_url(self, tmp_path: Path) -> None:
        """When hf_subdir is set, hf_hub_url receives the subdir-prefixed path."""
        s = HuggingFaceSource(
            format=ModelFormat.ONNX,
            hf_repo_id="org/repo",
            hf_subdir="mydir",
            files=["model.bin"],
            revision="rev",
        )

        def fake_dl(_url: str, dest: Path, **_kwargs: object) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("got")

        meta = mock.MagicMock()
        meta.size = 10

        with (
            mock.patch(
                "moment_to_action.models._sources._hugging_face.hf_hub_url",
                return_value="https://hf.co/file",
            ) as mock_url,
            mock.patch(
                "moment_to_action.models._sources._hugging_face.get_hf_file_metadata",
                return_value=meta,
            ),
            mock.patch(
                "moment_to_action.models._sources._hugging_face.download_file",
                side_effect=fake_dl,
            ),
        ):
            result = resolve_hugging_face_source(s, tmp_path, download=True, progress=False)

        assert result == tmp_path
        mock_url.assert_called_once_with(
            repo_id="org/repo", filename="mydir/model.bin", revision="rev"
        )

    def test_hf_subdir_preserves_local_structure(self, tmp_path: Path) -> None:
        """Nested files under hf_subdir are stored with relative structure in variant_dir."""
        s = HuggingFaceSource(
            format=ModelFormat.ONNX,
            hf_repo_id="org/repo",
            hf_subdir="mydir",
            files=["model.bin", "ref/out.bin"],
            revision="rev",
        )
        (tmp_path / "model.bin").write_text("ok")
        (tmp_path / "ref").mkdir()
        (tmp_path / "ref" / "out.bin").write_text("ok")

        with mock.patch("moment_to_action.models._sources._hugging_face.download_file") as mock_dl:
            result = resolve_hugging_face_source(s, tmp_path)

        assert result == tmp_path
        mock_dl.assert_not_called()
        assert (tmp_path / "model.bin").exists()
        assert (tmp_path / "ref" / "out.bin").exists()


# ---------------------------------------------------------------------------
# resolve_model_source dispatch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveModelSourceDispatch:
    """Tests for resolve_model_source dispatch by source type."""

    def test_dispatches_to_vendored(self) -> None:
        """A VendoredSource is routed through resolve_vendored_source."""
        s = VendoredSource(format=ModelFormat.ONNX, path=Path("yolo/model.onnx"))
        result = resolve_model_source(s, Path("/unused"))
        assert result is not None
        assert result.name == "model.onnx"

    def test_dispatches_to_download(self, tmp_path: Path) -> None:
        """A DownloadSource is routed through resolve_download_source."""
        s = DownloadSource(format=ModelFormat.ONNX, url="http://x", filename="m.bin")
        (tmp_path / "m.bin").write_text("ok")
        result = resolve_model_source(s, tmp_path)
        assert result == tmp_path / "m.bin"

    def test_dispatches_to_hugging_face(self, tmp_path: Path) -> None:
        """A HuggingFaceSource is routed through resolve_hugging_face_source."""
        s = HuggingFaceSource(
            format=ModelFormat.ONNX,
            hf_repo_id="org/repo",
            files=["a"],
            revision="r",
        )
        (tmp_path / "a").write_text("ok")
        result = resolve_model_source(s, tmp_path)
        assert result == tmp_path

    def test_dispatches_to_ultralytics(self, tmp_path: Path) -> None:
        """An UltralyticsSource is routed through resolve_ultralytics_source."""
        s = UltralyticsSource(format=ModelFormat.ONNX, name="yolov8n")
        (tmp_path / "model.onnx").write_text("ok")
        result = resolve_model_source(s, tmp_path)
        assert result == tmp_path / "model.onnx"

    def test_unsupported_source_raises(self, tmp_path: Path) -> None:
        """An unknown source type raises ValueError."""

        class BogusSource:
            pass

        with pytest.raises(ValueError, match="Unsupported ModelSource type"):
            resolve_model_source(BogusSource(), tmp_path)  # type: ignore[arg-type]
