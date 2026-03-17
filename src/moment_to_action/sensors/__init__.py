"""Sensor abstractions for the moment-to-action pipeline."""

from __future__ import annotations

from ._base import BaseSensor
from .file import FileSensor

__all__ = ["BaseSensor", "FileSensor"]
