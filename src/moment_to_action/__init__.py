"""Moment2Action project."""

import click
import rich.traceback
import rich_click

from ._version import VERSION
from .app import Moment2Action

# Fancy exceptions
rich.traceback.install(suppress=[click, rich_click])

# Set metadata
__version__ = VERSION

__all__ = ["Moment2Action", "__version__"]
