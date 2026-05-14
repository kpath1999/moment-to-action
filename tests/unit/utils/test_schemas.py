"""Unit tests for utils.schemas."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from moment_to_action.utils.schemas import update_frozen


class _Model(BaseModel, frozen=True):
    x: int = 0
    y: str = "hello"


@pytest.mark.unit
class TestUpdateFrozen:
    """Tests for update_frozen."""

    def test_returns_updated_field(self) -> None:
        """Updated field has the new value."""
        m = _Model(x=1, y="a")
        result = update_frozen(m, x=99)
        assert result.x == 99

    def test_unmentioned_fields_preserved(self) -> None:
        """Fields not in updates keep their original values."""
        m = _Model(x=1, y="a")
        result = update_frozen(m, x=2)
        assert result.y == "a"

    def test_returns_same_type(self) -> None:
        """Return type matches the input model type."""
        m = _Model()
        result = update_frozen(m, y="new")
        assert type(result) is _Model

    def test_original_unchanged(self) -> None:
        """Original frozen model is not mutated."""
        m = _Model(x=5)
        update_frozen(m, x=10)
        assert m.x == 5

    def test_multiple_fields_updated(self) -> None:
        """Multiple fields can be updated in one call."""
        m = _Model(x=1, y="old")
        result = update_frozen(m, x=2, y="new")
        assert result.x == 2
        assert result.y == "new"
