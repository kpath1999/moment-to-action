"""Root conftest for the test suite.

Custom markers:
- @pytest.mark.unit        — fast unit tests (run by default)
- @pytest.mark.integration — integration tests using real models (run by default)
- @pytest.mark.slow        — heavyweight tests skipped by default; use -m slow to include
"""

from __future__ import annotations

import sys

import pytest

# Pre-mock qairt at module level so it is in sys.modules before any sub-conftest
# or test module imports the project packages (which transitively import qairt).
# On real QCS6490 hardware the import succeeds and the mock is never installed.
if "qairt" not in sys.modules:
    try:
        import qairt  # noqa: F401
    except Exception:  # noqa: BLE001
        from unittest.mock import MagicMock

        sys.modules["qairt"] = MagicMock()


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: fast unit tests")
    config.addinivalue_line("markers", "integration: integration tests using real models")
    config.addinivalue_line("markers", "slow: heavyweight tests, skipped by default")
