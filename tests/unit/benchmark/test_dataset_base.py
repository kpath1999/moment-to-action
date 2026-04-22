from __future__ import annotations

import pytest

from moment_to_action.benchmark._datasets._base import BaseDataset


class _ToyDataset(BaseDataset[int]):
    @property
    def dataset_name(self) -> str:
        return "toy"

    def items(self) -> list[int]:
        return [1, 2, 3]


@pytest.mark.unit
def test_base_dataset_len_uses_items_length() -> None:
    dataset = _ToyDataset()
    assert len(dataset) == 3
