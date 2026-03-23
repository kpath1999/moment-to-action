"""Moment2Action project."""

import click
import rich.traceback
import rich_click

# Fancy exceptions
rich.traceback.install(suppress=[click, rich_click])
