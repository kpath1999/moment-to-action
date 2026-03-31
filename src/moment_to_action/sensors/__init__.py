"""Sensor abstractions for the moment-to-action pipeline."""

from __future__ import annotations

from ._base import BaseSensor
from ._camera_stream import CameraStreamSensor
from ._file_image import FileImageSensor

__all__ = ["BaseSensor", "CameraStreamSensor", "FileImageSensor"]
