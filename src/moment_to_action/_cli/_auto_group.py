import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast, overload

import rich_click as click
from rich_click import Group, RichGroup, command

_AnyCallable = Callable[..., Any]

PACKAGE_NAME = "moment_to_action"
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class AutoRichGroup(RichGroup):
    """RichGroup with automatic filesystem command loading."""

    def __init__(self, *args: Any, cmd_path: Path | None = None, **kwargs: Any) -> None:
        """Create a new automatic command group.

        Args:
            args:
                Positional arguments for click.
            cmd_path:
                Path to the directory containing commands.
            kwargs:
                Keyword arguments for click.
        """
        super().__init__(*args, **kwargs)

        if cmd_path:
            self.add_subcommands(cmd_path)

    def add_subcommands(self, dir_path: Path) -> None:
        """Add subcommands from a folder to click group.

        All command file/folders should be named cmd_X and export object X.
        """
        for path in dir_path.resolve().glob("cmd_*"):
            # Ensure files are python files
            if path.is_file() and path.suffix != ".py":
                continue

            # Get import path
            cmd_path = path if path.is_dir() else path.with_suffix("")
            cmd_name = cmd_path.name[4:]

            project_path = cmd_path.relative_to(PROJECT_ROOT)
            import_path = PACKAGE_NAME + "." + str(project_path).replace("/", ".")

            # Import that object
            mod = importlib.import_module(import_path)
            cmd_obj = getattr(mod, cmd_name, None)

            if cmd_obj and callable(cmd_obj) and isinstance(cmd_obj, click.Command):
                self.add_command(
                    cmd_obj,
                    aliases=getattr(cmd_obj, "aliases", None),
                    panel=getattr(cmd_obj, "panel", None),
                )
            else:
                msg = f'Command "{cmd_name}" does not exist in "{import_path}"'
                raise RuntimeError(msg)


# variant: no call, directly as decorator for a function.
@overload
def auto_group(name: _AnyCallable) -> RichGroup: ...


# variant: with positional name and with positional or keyword cls argument:
# @group(namearg, GroupCls, ...) or @group(namearg, cls=GroupCls, ...)
@overload
def auto_group[G: Group](
    name: str | None,
    cls: type[G],
    **attrs: Any,
) -> Callable[[_AnyCallable], G]: ...


# variant: name omitted, cls _must_ be a keyword argument, @group(cmd=GroupCls, ...)
@overload
def auto_group[G: Group](
    name: None = None,
    *,
    cls: type[G],
    **attrs: Any,
) -> Callable[[_AnyCallable], G]: ...


# variant: with optional string name, no cls argument provided.
@overload
def auto_group(
    name: str | None = ...,
    cls: None = None,
    **attrs: Any,
) -> Callable[[_AnyCallable], RichGroup]: ...


def auto_group[G: Group](
    name: str | _AnyCallable | None = None,
    cls: type[G] | None = None,
    **attrs: Any,
) -> Group | Callable[[_AnyCallable], RichGroup | G]:
    """Group decorator function.

    Defines the group() function so that it uses the AutoRichGroup class by default.
    """
    if cls is None:
        cls = cast("type[G]", AutoRichGroup)

    if callable(name):
        return command(cls=cls, **attrs)(name)

    return command(name, cls, **attrs)
