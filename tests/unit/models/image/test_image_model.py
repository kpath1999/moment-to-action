"""Unit tests for ImageModel ABC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moment_to_action.models.image._base import ImageModel


@pytest.mark.unit
class TestImageModel:
    """Tests for ImageModel abstract base class."""

    def test_cannot_instantiate_abstract(self) -> None:
        """ImageModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ImageModel("default", Path("/x"))  # type: ignore[abstract]

    def test_abstract_methods_enforced(self) -> None:
        """Subclasses that skip abstract methods cannot be instantiated."""

        class _Incomplete(ImageModel):
            def load(self, backend: object) -> None:
                """Load."""

            def unload(self) -> None:
                """Unload."""

            def prepare(self, frame: np.ndarray) -> np.ndarray:
                """Prepare."""
                return frame

            # Missing run and post_proc

        with pytest.raises(TypeError):
            _Incomplete("v", Path("/x"))  # type: ignore[abstract]
