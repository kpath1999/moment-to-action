from __future__ import annotations

import json
import os
import random
from pathlib import Path

import attrs
import platformdirs

from moment_to_action.benchmark._datasets._base import BaseDataset

_DEFAULT_DATASET_ID = "lmms-lab/MSRVTT-QA"
_DEFAULT_SPLIT = "test"
_SCAN_MULTIPLIER = 20


def _default_cache_dir() -> Path:
    return platformdirs.user_cache_path("moment_to_action", "GATech") / "msrvtt_qa"


@attrs.frozen
class MsrvttItem:
    """One MSRVTT-QA example used for benchmark evaluation."""

    video_path: Path
    question: str
    answer: str


@attrs.define
class MsrvttDataset(BaseDataset[MsrvttItem]):
    """MSRVTT-QA loader backed by the HuggingFace datasets library."""

    n_items: int = 500
    cache_dir: Path = attrs.Factory(_default_cache_dir)
    seed: int = 42
    dataset_id: str = _DEFAULT_DATASET_ID
    split: str = _DEFAULT_SPLIT
    local_dir: Path | None = None
    _items: list[MsrvttItem] = attrs.field(factory=list, init=False)

    def __attrs_post_init__(self) -> None:
        if self.n_items <= 0:
            msg = "n_items must be greater than 0"
            raise ValueError(msg)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._items = self._load_local_items() if self.local_dir is not None else self._load_items()

    @property
    def dataset_name(self) -> str:
        """Dataset identifier used in benchmark output payloads."""
        return "msrvtt_qa"

    def items(self) -> list[MsrvttItem]:
        """Return sampled QA items for evaluation."""
        return list(self._items)

    def _load_items(self) -> list[MsrvttItem]:  # noqa: C901
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - guarded by dependency
            msg = "datasets package is required for MsrvttDataset"
            raise RuntimeError(msg) from exc

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        split_candidates = [self.split, "test", "validation", "train"]

        parsed: list[MsrvttItem] = []
        last_exc: Exception | None = None
        for split in dict.fromkeys(split_candidates):
            try:
                dataset = load_dataset(
                    self.dataset_id,
                    split=split,
                    cache_dir=str(self.cache_dir),
                    streaming=True,
                    token=token,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue

            max_rows_to_scan = max(self.n_items * _SCAN_MULTIPLIER, self.n_items)
            for row_idx, row in enumerate(dataset):
                if row_idx >= max_rows_to_scan:
                    break
                maybe_item = self._parse_row(row)
                if maybe_item is not None:
                    parsed.append(maybe_item)

            if parsed:
                break

        if not parsed:
            hint = ""
            if last_exc is not None:
                text = str(last_exc)
                if "404" in text or "DatasetNotFoundError" in text or "doesn't exist" in text:
                    hint = (
                        " Dataset id was not found on the Hub. Pass a valid id via "
                        "--msrvtt-dataset-id (and optionally --msrvtt-split)."
                    )
                elif "401" in text or "cannot be accessed" in text:
                    hint = (
                        " Dataset access appears restricted; set HF_TOKEN (or "
                        "HUGGING_FACE_HUB_TOKEN) with a token that has access."
                    )
            msg = f"Unable to load usable MSRVTT items from '{self.dataset_id}'.{hint}"
            raise RuntimeError(msg) from last_exc

        sample_size = min(self.n_items, len(parsed))
        rng = random.Random(self.seed)  # noqa: S311
        return rng.sample(parsed, sample_size)

    @staticmethod
    def _parse_row(row: object) -> MsrvttItem | None:
        if not isinstance(row, dict):
            return None

        question = MsrvttDataset._as_str(row.get("question"))
        answer = MsrvttDataset._as_str(row.get("answer"))
        video_path = MsrvttDataset._extract_video_path(row)

        if not question or not answer or video_path is None:
            return None

        return MsrvttItem(video_path=video_path, question=question, answer=answer)

    @staticmethod
    def _extract_video_path(row: dict[str, object]) -> Path | None:
        for key in ("video_path", "video", "video_file"):
            value = row.get(key)
            if isinstance(value, str) and value:
                return Path(value)
            if isinstance(value, dict):
                path_value = value.get("path")
                if isinstance(path_value, str) and path_value:
                    return Path(path_value)
        return None

    @staticmethod
    def _as_str(value: object) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
        return ""

    def _load_local_items(self) -> list[MsrvttItem]:
        assert self.local_dir is not None  # noqa: S101 — only called when local_dir is set
        local_dir = self.local_dir

        qa_path = local_dir / f"{self.split}_qa.json"
        if not qa_path.exists():
            msg = (
                f"QA annotation file not found: {qa_path}. "
                f"Expected '{self.split}_qa.json' inside the local directory."
            )
            raise RuntimeError(msg)

        with qa_path.open(encoding="utf-8") as fh:
            entries = json.load(fh)

        if not isinstance(entries, list):
            msg = f"Expected a JSON list in {qa_path}, got {type(entries).__name__}"
            raise TypeError(msg)

        videos_dir = local_dir / "videos"
        parsed: list[MsrvttItem] = []
        for entry in entries:
            item = self._parse_local_entry(entry, videos_dir)
            if item is not None:
                parsed.append(item)

        if not parsed:
            msg = f"No usable QA items found in {qa_path} (looked for videos in {videos_dir})."
            raise RuntimeError(msg)

        sample_size = min(self.n_items, len(parsed))
        rng = random.Random(self.seed)  # noqa: S311
        return rng.sample(parsed, sample_size)

    @staticmethod
    def _parse_local_entry(entry: object, videos_dir: Path) -> MsrvttItem | None:
        """Parse one entry from a local MSRVTT-QA JSON annotation file."""
        if not isinstance(entry, dict):
            return None
        video_id = entry.get("video_id")
        question = MsrvttDataset._as_str(entry.get("question"))
        answer = MsrvttDataset._as_str(entry.get("answer"))
        if video_id is None or not question or not answer:
            return None
        video_path = videos_dir / f"video{video_id}.mp4"
        if not video_path.exists():
            return None
        return MsrvttItem(video_path=video_path, question=question, answer=answer)
