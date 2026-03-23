import logging

import click
import rich_click
from rich.console import Console
from rich.logging import RichHandler

_stderr_console = Console(stderr=True)


def init_logging(*, verbose: bool) -> None:
    """Initialize logging.

    Args:
        verbose:
            Should verbose messages be printed?
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_suppress=[click, rich_click],
                console=_stderr_console,
            ),
        ],
    )
