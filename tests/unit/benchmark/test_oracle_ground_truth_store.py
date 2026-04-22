from __future__ import annotations

from pathlib import Path

import pytest

from moment_to_action.benchmark._oracle_ground_truth import OracleStore


@pytest.mark.unit
def test_oracle_store_path_property_uses_explicit_path(tmp_path: Path) -> None:
    path = tmp_path / "oracle.json"
    store = OracleStore(path=path)
    assert store.path == path
