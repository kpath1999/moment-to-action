"""Moment2Action project."""

import click
import rich.traceback
import rich_click

from ._version import VERSION

# Fancy exceptions
rich.traceback.install(suppress=[click, rich_click])

# Set metadata
__version__ = VERSION
