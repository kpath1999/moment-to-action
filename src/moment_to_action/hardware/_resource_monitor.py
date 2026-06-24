"""ResourceMonitor — abstract base for platform resource sampling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from moment_to_action.hardware._types import ComputeUnit, ComputeUnitUsageSample


class ResourceMonitor(ABC):
    """Abstract resource monitor.

    Reads power draw and utilisation for a given compute unit.  Concrete
    subclasses live in ``hardware/_platforms/<chip>/_resources.py``.
    """

    @abstractmethod
    def sample(self, unit: ComputeUnit) -> ComputeUnitUsageSample:
        """Return a resource measurement for *unit*.

        Args:
            unit: The compute unit to sample.

        Returns:
            A ``ComputeUnitUsageSample`` with current power and utilisation figures.
        """
        ...

    @staticmethod
    def used_memory_mb() -> float:
        """Return used system memory in megabytes.

        Uses ``total - available`` per psutil docs — more accurate than ``.used``
        because ``.available`` accounts for reclaimable cache pages.

        Returns:
            System memory used in megabytes.
        """
        vm = psutil.virtual_memory()
        return (vm.total - vm.available) / (1024 * 1024)
