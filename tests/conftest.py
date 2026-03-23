"""Root conftest for the test suite.

Custom markers:
- @pytest.mark.unit        — fast unit tests (run by default)
- @pytest.mark.integration — integration tests using real models (run by default)
- @pytest.mark.slow        — heavyweight tests skipped by default; use -m slow to include
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: fast unit tests")
    config.addinivalue_line("markers", "integration: integration tests using real models")
    config.addinivalue_line("markers", "slow: heavyweight tests, skipped by default")
