"""Root conftest for the test suite.

Custom markers:
- @pytest.mark.unit        — fast unit tests (run by default)
- @pytest.mark.integration — integration tests using real models (run by default)
- @pytest.mark.slow        — heavyweight tests skipped by default; use -m slow to include
"""

from __future__ import annotations

import sys

import pytest

# qairt-dev installs Python bindings but requires the QAIRT SDK at
# /opt/qcom/aistack/qairt/<version>/. Dev machines have the full SDK so the
# import succeeds. CI/test machines do not, so we mock the module to allow the
# suite to run without hardware.
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
