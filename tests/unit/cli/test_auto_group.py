"""Unit tests for _cli/_auto_group.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich_click import RichGroup

from moment_to_action._cli._auto_group import AutoRichGroup, auto_group

_COMMANDS_DIR = (
    Path(__file__).parent.parent.parent.parent / "src" / "moment_to_action" / "_cli" / "commands"
)


@pytest.mark.unit
class TestAutoRichGroup:
    """Tests for AutoRichGroup automatic filesystem command loading."""

    def test_init_no_cmd_path_adds_no_commands(self) -> None:
        """No cmd_path → no subcommands registered."""
        group = AutoRichGroup(name="test")
        assert group.commands == {}

    def test_add_subcommands_skips_non_python_files(self, tmp_path: Path) -> None:
        """Non-.py cmd_* files are silently ignored."""
        (tmp_path / "cmd_thing.txt").touch()
        group = AutoRichGroup(name="test")
        group.add_subcommands(tmp_path)
        assert group.commands == {}

    def test_add_subcommands_loads_real_command(self) -> None:
        """AutoRichGroup with the actual commands dir loads read_power."""
        group = AutoRichGroup(name="test", cmd_path=_COMMANDS_DIR)
        assert "read-power" in group.commands

    def test_add_subcommands_missing_export_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """RuntimeError when the module has no attribute matching the command name."""
        import moment_to_action._cli._auto_group as ag_module

        (tmp_path / "cmd_foo.py").touch()
        monkeypatch.setattr(ag_module, "PROJECT_ROOT", tmp_path)

        fake_module = MagicMock(spec=[])  # no attributes at all
        monkeypatch.setattr(ag_module.importlib, "import_module", lambda _: fake_module)

        group = AutoRichGroup(name="test")
        with pytest.raises(RuntimeError, match='Command "foo"'):
            group.add_subcommands(tmp_path)

    def test_add_subcommands_non_click_object_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """RuntimeError when the module exports a non-Click object for the command name."""
        import moment_to_action._cli._auto_group as ag_module

        (tmp_path / "cmd_bar.py").touch()
        monkeypatch.setattr(ag_module, "PROJECT_ROOT", tmp_path)

        fake_module = MagicMock()
        fake_module.bar = "not a click command"
        monkeypatch.setattr(ag_module.importlib, "import_module", lambda _: fake_module)

        group = AutoRichGroup(name="test")
        with pytest.raises(RuntimeError, match='Command "bar"'):
            group.add_subcommands(tmp_path)


@pytest.mark.unit
class TestAutoGroupDecorator:
    """Tests for the auto_group decorator (overload variants)."""

    def test_direct_decorator(self) -> None:
        """@auto_group used without parentheses wraps the function."""

        @auto_group
        def mygroup() -> None:
            """My group."""

        assert isinstance(mygroup, RichGroup)

    def test_decorator_with_name(self) -> None:
        """@auto_group('name') creates a named group."""

        @auto_group("custom")
        def mygroup() -> None:
            """My group."""

        assert isinstance(mygroup, RichGroup)
        assert mygroup.name == "custom"

    def test_decorator_with_custom_cls(self) -> None:
        """@auto_group(cls=RichGroup) uses the provided class, not AutoRichGroup."""

        @auto_group(cls=RichGroup)
        def mygroup() -> None:
            """My group."""

        assert isinstance(mygroup, RichGroup)
        assert not isinstance(mygroup, AutoRichGroup)

    def test_decorator_default_cls_is_auto_rich_group(self) -> None:
        """Without explicit cls=, the created group is an AutoRichGroup."""

        @auto_group()
        def mygroup() -> None:
            """My group."""

        assert isinstance(mygroup, AutoRichGroup)
