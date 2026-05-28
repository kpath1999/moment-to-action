"""Unit tests for ImageDetectionModel ABC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moment_to_action.models.image.detection._base import ImageDetectionModel


@pytest.mark.unit
class TestImageDetectionModel:
    """Tests for ImageDetectionModel abstract base class."""

    def test_cannot_instantiate_abstract(self) -> None:
        """ImageDetectionModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ImageDetectionModel("default", Path("/x"))  # type: ignore[abstract]

    def test_abstract_methods_enforced(self) -> None:
        """Subclasses missing abstract methods cannot be instantiated."""

        class _Incomplete(ImageDetectionModel):
            def load(self, backend: object) -> None:
                """Load."""

            def unload(self) -> None:
                """Unload."""

            def prepare(self, frame: np.ndarray) -> np.ndarray:
                """Prepare."""
                return frame

            def run(self, prepared: np.ndarray) -> object:
                """Run."""
                return prepared

            # Missing post_proc

        with pytest.raises(TypeError):
            _Incomplete("v", Path("/x"))  # type: ignore[abstract]
