"""Unit tests for ImageStage base class."""

from __future__ import annotations

import pytest

from moment_to_action.stages._base import Stage
from moment_to_action.stages.image._base import ImageStage


@pytest.mark.unit
class TestImageStage:
    """Tests for ImageStage marker base class."""

    def test_image_stage_is_subclass_of_stage(self) -> None:
        """ImageStage must extend Stage."""
        assert issubclass(ImageStage, Stage)

    def test_image_stage_is_abstract(self) -> None:
        """ImageStage cannot be instantiated directly (no _process implementation)."""
        with pytest.raises(TypeError):
            ImageStage()  # type: ignore[abstract]
