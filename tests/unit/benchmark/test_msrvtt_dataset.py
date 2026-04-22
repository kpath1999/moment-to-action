from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from moment_to_action.benchmark._datasets._msrvtt_dataset import MsrvttDataset


@pytest.mark.unit
def test_msrvtt_dataset_parse_row_happy_path() -> None:
    row = {
        "question": "What is happening?",
        "answer": "A man is running",
        "video": {"path": "/tmp/sample.mp4"},
    }

    item = MsrvttDataset._parse_row(row)

    assert item is not None
    assert item.video_path == Path("/tmp/sample.mp4")
    assert item.question == "What is happening?"
    assert item.answer == "A man is running"


@pytest.mark.unit
def test_msrvtt_dataset_parse_row_non_dict_returns_none() -> None:
    assert MsrvttDataset._parse_row("not-a-dict") is None


@pytest.mark.unit
def test_msrvtt_dataset_rejects_non_positive_n_items(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MsrvttDataset, "_load_items", lambda _self: [])
    with pytest.raises(ValueError, match="n_items must be greater than 0"):
        MsrvttDataset(n_items=0)


# ---------------------------------------------------------------------------
# Local directory loading
# ---------------------------------------------------------------------------


def _write_qa(path: Path, entries: list[object]) -> None:
    path.write_text(json.dumps(entries), encoding="utf-8")


@pytest.mark.unit
def test_parse_local_entry_happy_path(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "video42.mp4").touch()

    entry = {"video_id": 42, "question": "What is happening?", "answer": "A cat jumped"}
    item = MsrvttDataset._parse_local_entry(entry, videos_dir)

    assert item is not None
    assert item.video_path == videos_dir / "video42.mp4"
    assert item.question == "What is happening?"
    assert item.answer == "A cat jumped"


@pytest.mark.unit
def test_parse_local_entry_missing_video(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()

    entry = {"video_id": 99, "question": "Q?", "answer": "A"}
    assert MsrvttDataset._parse_local_entry(entry, videos_dir) is None


@pytest.mark.unit
def test_parse_local_entry_non_dict(tmp_path: Path) -> None:
    assert MsrvttDataset._parse_local_entry("not a dict", tmp_path) is None


@pytest.mark.unit
def test_parse_local_entry_missing_question_answer(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "video1.mp4").touch()

    assert MsrvttDataset._parse_local_entry({"video_id": 1}, videos_dir) is None


@pytest.mark.unit
def test_msrvtt_dataset_local_dir_loads_items(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "video0.mp4").touch()
    (videos_dir / "video1.mp4").touch()

    _write_qa(
        tmp_path / "test_qa.json",
        [
            {"video_id": 0, "question": "Q1?", "answer": "A1"},
            {"video_id": 1, "question": "Q2?", "answer": "A2"},
            {"video_id": 99, "question": "Q3?", "answer": "A3"},  # no video — skipped
        ],
    )

    dataset = MsrvttDataset(n_items=10, local_dir=tmp_path)

    assert len(dataset.items()) == 2
    assert {i.question for i in dataset.items()} == {"Q1?", "Q2?"}


@pytest.mark.unit
def test_msrvtt_dataset_local_dir_respects_n_items(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    for vid_id in range(5):
        (videos_dir / f"video{vid_id}.mp4").touch()

    _write_qa(
        tmp_path / "test_qa.json",
        [{"video_id": i, "question": f"Q{i}?", "answer": f"A{i}"} for i in range(5)],
    )

    dataset = MsrvttDataset(n_items=3, local_dir=tmp_path)
    assert len(dataset.items()) == 3


@pytest.mark.unit
def test_msrvtt_dataset_local_dir_missing_json(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="QA annotation file not found"):
        MsrvttDataset(n_items=10, local_dir=tmp_path)


@pytest.mark.unit
def test_msrvtt_dataset_local_dir_invalid_json_format(tmp_path: Path) -> None:
    (tmp_path / "test_qa.json").write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    with pytest.raises(TypeError, match="Expected a JSON list"):
        MsrvttDataset(n_items=10, local_dir=tmp_path)


@pytest.mark.unit
def test_msrvtt_dataset_local_dir_no_usable_items(tmp_path: Path) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    _write_qa(
        tmp_path / "test_qa.json",
        [{"video_id": 99, "question": "Q?", "answer": "A"}],  # no matching video
    )

    with pytest.raises(RuntimeError, match="No usable QA items"):
        MsrvttDataset(n_items=10, local_dir=tmp_path)


@pytest.mark.unit
def test_msrvtt_as_str_and_extract_video_path_helpers() -> None:
    assert MsrvttDataset._as_str(" hello ") == "hello"
    assert MsrvttDataset._as_str(["", " ok "]) == "ok"
    assert MsrvttDataset._as_str(1) == ""

    assert MsrvttDataset._extract_video_path({"video_path": "/tmp/a.mp4"}) == Path("/tmp/a.mp4")
    assert MsrvttDataset._extract_video_path({"video": {"path": "/tmp/b.mp4"}}) == Path(
        "/tmp/b.mp4"
    )
    assert MsrvttDataset._extract_video_path({"video": {"path": 1}}) is None


@pytest.mark.unit
def test_msrvtt_load_items_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_rows = [
        {"question": "Q1", "answer": "A1", "video": {"path": "/tmp/v1.mp4"}},
        {"question": "Q2", "answer": "A2", "video_path": "/tmp/v2.mp4"},
        {"question": "bad", "answer": "", "video_path": "/tmp/v3.mp4"},
    ]

    fake_datasets = mock.MagicMock()
    fake_datasets.load_dataset.return_value = iter(fake_rows)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    dataset = MsrvttDataset(n_items=2, cache_dir=tmp_path, split="test")
    assert len(dataset.items()) == 2
    assert dataset.dataset_name == "msrvtt_qa"


@pytest.mark.unit
def test_msrvtt_load_items_404_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_datasets = mock.MagicMock()
    fake_datasets.load_dataset.side_effect = Exception("404 DatasetNotFoundError")
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(RuntimeError, match="Dataset id was not found"):
        MsrvttDataset(n_items=1, cache_dir=tmp_path)


@pytest.mark.unit
def test_msrvtt_load_items_401_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_datasets = mock.MagicMock()
    fake_datasets.load_dataset.side_effect = Exception("401 cannot be accessed")
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        MsrvttDataset(n_items=1, cache_dir=tmp_path)


@pytest.mark.unit
def test_msrvtt_load_items_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace())

    with pytest.raises(RuntimeError, match="datasets package is required"):
        MsrvttDataset(n_items=1)


@pytest.mark.unit
def test_msrvtt_load_items_respects_scan_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_datasets = mock.MagicMock()
    fake_datasets.load_dataset.return_value = ({"bad": i} for i in range(1000))
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    parse_calls = {"count": 0}

    def _count_parse(_row: object) -> None:
        parse_calls["count"] += 1

    monkeypatch.setattr(MsrvttDataset, "_parse_row", staticmethod(_count_parse))

    with pytest.raises(RuntimeError, match="Unable to load usable MSRVTT items"):
        MsrvttDataset(n_items=1, cache_dir=tmp_path)

    # n_items=1 -> max_rows_to_scan=max(1*20,1)=20, attempted across 3 fallbacks.
    assert parse_calls["count"] == 60
