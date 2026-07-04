"""Unit tests for prompting._grammar."""

from __future__ import annotations

import pytest

from moment_to_action.prompting import YES_NO_GRAMMAR


@pytest.mark.unit
class TestYesNoGrammar:
    """Tests for YES_NO_GRAMMAR."""

    def test_is_nonempty_string(self) -> None:
        """YES_NO_GRAMMAR is a non-empty string."""
        assert isinstance(YES_NO_GRAMMAR, str)
        assert len(YES_NO_GRAMMAR) > 0

    def test_forces_leading_yes_or_no(self) -> None:
        """The grammar's root rule constrains the leading token to YES or NO."""
        assert '"YES"' in YES_NO_GRAMMAR
        assert '"NO"' in YES_NO_GRAMMAR

    def test_has_root_rule(self) -> None:
        """The grammar defines a root rule, as required by GBNF."""
        assert "root ::=" in YES_NO_GRAMMAR
