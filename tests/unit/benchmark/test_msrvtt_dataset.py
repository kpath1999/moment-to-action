from __future__ import annotations

import json
from pathlib import Path

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
