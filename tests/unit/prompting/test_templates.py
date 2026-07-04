"""Unit tests for prompting._templates."""

from __future__ import annotations

import pytest

from moment_to_action.prompting import BENCHMARK_SYSTEM, CHATML, PHI3


@pytest.mark.unit
class TestTemplates:
    """Tests for the chat template format strings."""

    def test_chatml_formats_system_and_user(self) -> None:
        """CHATML substitutes {system} and {user} placeholders."""
        result = CHATML.format(system="sys msg", user="user msg")
        assert "sys msg" in result
        assert "user msg" in result
        assert "<|im_start|>assistant" in result

    def test_phi3_formats_system_and_user(self) -> None:
        """PHI3 substitutes {system} and {user} placeholders."""
        result = PHI3.format(system="sys msg", user="user msg")
        assert "sys msg" in result
        assert "user msg" in result
        assert "<|assistant|>" in result

    def test_benchmark_system_is_nonempty_string(self) -> None:
        """BENCHMARK_SYSTEM is a non-empty string."""
        assert isinstance(BENCHMARK_SYSTEM, str)
        assert len(BENCHMARK_SYSTEM) > 0
