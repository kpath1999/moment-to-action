"""Unit tests for _cli/_params.py."""

from __future__ import annotations

import pytest

from moment_to_action._cli._params import BASED_INT, BasedIntParamType


@pytest.mark.unit
class TestBasedIntParamType:
    """Tests for BasedIntParamType — multi-base integer click parameter."""

    def test_int_passthrough(self) -> None:
        """Already-int values are returned unchanged."""
        assert BASED_INT.convert(42, None, None) == 42

    def test_decimal_string(self) -> None:
        """Decimal string converts to int."""
        assert BASED_INT.convert("42", None, None) == 42

    def test_hex_string(self) -> None:
        """Hex string (0x…) converts to int."""
        assert BASED_INT.convert("0x1a", None, None) == 26

    def test_octal_string(self) -> None:
        """Octal string (0o…) converts to int."""
        assert BASED_INT.convert("0o17", None, None) == 15

    def test_binary_string(self) -> None:
        """Binary string (0b…) converts to int."""
        assert BASED_INT.convert("0b1010", None, None) == 10

    def test_invalid_string_fails(self) -> None:
        """Non-numeric string raises BadParameter."""
        import click as _click

        with pytest.raises(_click.exceptions.BadParameter):
            BASED_INT.convert("notanumber", None, None)

    def test_name_attribute(self) -> None:
        """Type is advertised as 'integer'."""
        assert BasedIntParamType.name == "integer"
