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
            ImageModel("default", Path("/x"))  # type: ignore[abstract, call-arg]

    def test_abstract_methods_enforced(self) -> None:
        """Subclasses that skip abstract methods cannot be instantiated."""
        from moment_to_action.hardware import Platform
        from moment_to_action.hardware._types import ComputeUnit

        class _Incomplete(ImageModel[object, object]):
            def load(self, platform: Platform, unit: ComputeUnit) -> None:
                """Load."""

            def unload(self) -> None:
                """Unload."""

            def prepare(self, inputs: np.ndarray) -> np.ndarray:
                """Prepare."""
                return inputs

            # Missing run, post_proc, verify_outputs

        with pytest.raises(TypeError):
            _Incomplete("v", Path("/x"))  # type: ignore[abstract, call-arg]

    def test_verify_outputs_abstract_enforced(self) -> None:
        """Subclass missing verify_outputs cannot be instantiated."""

        class _NoVerify(ImageModel[object, object]):
            def load(self, backend: object, unit: object = None) -> None:  # type: ignore[override]
                """Load."""

            def unload(self) -> None:
                """Unload."""

            def prepare(self, inputs: np.ndarray) -> np.ndarray:
                """Prepare."""
                return inputs

            def run(self, prepared: np.ndarray) -> object:
                """Run."""
                return prepared

            def post_proc(self, raw: object) -> list[object]:
                """Post-process."""
                return []

            # Missing verify_outputs

        with pytest.raises(TypeError):
            _NoVerify("v", Path("/x"))  # type: ignore[abstract, call-arg]
