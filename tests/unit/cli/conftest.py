"""Shared fixtures for cli unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_backend() -> MagicMock:
    """Return a MagicMock ComputeBackend with a usable power monitor."""
    sample = MagicMock()
    sample.power_mw = 100
    sample.utilization_pct = 10
    pwr_mon = MagicMock()
    pwr_mon.sample.return_value = sample
    backend = MagicMock()
    backend.power_monitor = pwr_mon
    return backend
