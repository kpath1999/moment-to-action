import logging

import click
import rich_click
from rich.console import Console
from rich.logging import RichHandler

_stderr_console = Console(stderr=True)


def init_logging(*, log_level: str) -> None:
    """Initialize logging.

    Args:
        log_level:
            Logging level name (e.g. ``"DEBUG"``, ``"INFO"``).
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
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

    # Fix loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
