"""Shared fixtures for cli unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_backend() -> MagicMock:
    """Return a MagicMock Platform with a usable power monitor."""
    sample = MagicMock()
    sample.power_mw = 100
    sample.usage_pct = 10
    resource_mon = MagicMock()
    resource_mon.sample.return_value = sample
    backend = MagicMock()
    backend.resource_monitor = resource_mon
    return backend
