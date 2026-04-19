from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

ItemT = TypeVar("ItemT")


class BaseDataset[ItemT](ABC):
    """Common interface for benchmark datasets."""

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Return a stable dataset identifier for reporting."""

    @abstractmethod
    def items(self) -> list[ItemT]:
        """Return evaluation items from the dataset."""

    def __len__(self) -> int:
        """Return the number of currently selected evaluation items."""
        return len(self.items())
