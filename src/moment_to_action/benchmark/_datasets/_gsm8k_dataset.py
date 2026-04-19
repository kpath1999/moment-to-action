from __future__ import annotations

import random
import re
from pathlib import Path  # noqa: TC003

import attrs
import platformdirs

from moment_to_action.benchmark._datasets._base import BaseDataset

_DEFAULT_DATASET_ID = "openai/gsm8k"
_DEFAULT_CONFIG = "main"
_DEFAULT_SPLIT = "test"
_ANS_MARKER = "####"
_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def _default_cache_dir() -> Path:
    return platformdirs.user_cache_path("moment_to_action", "GATech") / "gsm8k"


@attrs.frozen
class GSM8KItem:
    """One GSM8K problem with normalized numeric ground-truth answer."""

    question: str
    answer: str


@attrs.define
class GSM8KDataset(BaseDataset[GSM8KItem]):
    """GSM8K test loader backed by the HuggingFace datasets library."""

    n_items: int = 500
    cache_dir: Path = attrs.Factory(_default_cache_dir)
    seed: int = 42
    dataset_id: str = _DEFAULT_DATASET_ID
    config_name: str = _DEFAULT_CONFIG
    split: str = _DEFAULT_SPLIT
    _items: list[GSM8KItem] = attrs.field(factory=list, init=False)

    def __attrs_post_init__(self) -> None:
        if self.n_items <= 0:
            msg = "n_items must be greater than 0"
            raise ValueError(msg)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._items = self._load_items()

    @property
    def dataset_name(self) -> str:
        """Dataset identifier used in benchmark output payloads."""
        return "gsm8k_test"

    def items(self) -> list[GSM8KItem]:
        """Return sampled GSM8K test items."""
        return list(self._items)

    def _load_items(self) -> list[GSM8KItem]:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - guarded by dependency
            msg = "datasets package is required for GSM8KDataset"
            raise RuntimeError(msg) from exc

        dataset = load_dataset(
            self.dataset_id,
            self.config_name,
            split=self.split,
            cache_dir=str(self.cache_dir),
        )

        parsed: list[GSM8KItem] = []
        for row in dataset:
            item = self._parse_row(row)
            if item is not None:
                parsed.append(item)

        if not parsed:
            msg = "No valid GSM8K items were parsed from dataset rows"
            raise RuntimeError(msg)

        sample_size = min(self.n_items, len(parsed))
        rng = random.Random(self.seed)  # noqa: S311
        return rng.sample(parsed, sample_size)

    @staticmethod
    def _parse_row(row: object) -> GSM8KItem | None:
        if not isinstance(row, dict):
            return None

        question = row.get("question")
        answer_raw = row.get("answer")
        if not isinstance(question, str) or not isinstance(answer_raw, str):
            return None

        normalized_answer = GSM8KDataset.extract_numeric_answer(answer_raw)
        if normalized_answer is None:
            return None

        return GSM8KItem(question=question.strip(), answer=normalized_answer)

    @staticmethod
    def extract_numeric_answer(text: str) -> str | None:
        """Extract and normalize a numeric answer from GSM8K answer text."""
        candidate = text
        if _ANS_MARKER in text:
            candidate = text.rsplit(_ANS_MARKER, maxsplit=1)[-1]

        matches = _NUMBER_RE.findall(candidate)
        if not matches and candidate is not text:
            matches = _NUMBER_RE.findall(text)
        if not matches:
            return None

        return GSM8KDataset.normalize_number(matches[-1])

    @staticmethod
    def normalize_number(value: str) -> str:
        """Normalize a numeric string for exact-match comparison."""
        compact = value.replace(",", "").strip()
        try:
            numeric = float(compact)
        except ValueError:
            return compact

        if numeric.is_integer():
            return str(int(numeric))
        return str(numeric)
