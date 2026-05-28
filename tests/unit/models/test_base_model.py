"""Unit tests for BaseModel ABC."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from moment_to_action.models._base import BaseModel


class _ConcreteModel(BaseModel):
    """Minimal concrete subclass for testing."""

    def load(self, backend: object) -> None:
        """Load the model."""
        self._backend = backend  # type: ignore[assignment]

    def unload(self) -> None:
        """Unload the model."""
        self._backend = None


@pytest.mark.unit
class TestBaseModel:
    """Tests for BaseModel abstract base class."""

    def test_cannot_instantiate_abstract(self) -> None:
        """BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel("default", Path("/x"))  # type: ignore[abstract]

    def test_init_stores_variant_and_path(self) -> None:
        """__init__ stores _variant and _path."""
        model = _ConcreteModel("myvariant", Path("/some/path.onnx"))
        assert model._variant == "myvariant"
        assert model._path == Path("/some/path.onnx")

    def test_backend_starts_as_none(self) -> None:
        """_backend is None before load() is called."""
        model = _ConcreteModel("default", Path("/x"))
        assert model._backend is None

    def test_load_sets_backend(self) -> None:
        """load() sets _backend."""
        model = _ConcreteModel("default", Path("/x"))
        mock_backend = MagicMock()
        model.load(mock_backend)
        assert model._backend is mock_backend

    def test_unload_clears_backend(self) -> None:
        """unload() clears _backend."""
        model = _ConcreteModel("default", Path("/x"))
        model._backend = MagicMock()
        model.unload()
        assert model._backend is None


@pytest.mark.unit
class TestBaseModelIsLoaded:
    """Tests for the is_loaded property."""

    def test_is_loaded_false_before_load(self) -> None:
        """is_loaded is False before load() is called."""
        model = _ConcreteModel("default", Path("/x"))
        assert model.is_loaded is False

    def test_is_loaded_true_after_load(self) -> None:
        """is_loaded is True after load() is called."""
        model = _ConcreteModel("default", Path("/x"))
        model.load(MagicMock())
        assert model.is_loaded is True

    def test_is_loaded_false_after_unload(self) -> None:
        """is_loaded is False after unload() is called."""
        model = _ConcreteModel("default", Path("/x"))
        model.load(MagicMock())
        model.unload()
        assert model.is_loaded is False


@pytest.mark.unit
class TestBaseModelLoadedContextManager:
    """Tests for the loaded() context manager."""

    def test_loaded_enters_and_yields_self(self) -> None:
        """loaded() yields the model itself."""
        model = _ConcreteModel("default", Path("/x"))
        backend = MagicMock()
        with model.loaded(backend) as m:
            assert m is model
            assert model.is_loaded is True

    def test_loaded_unloads_on_clean_exit(self) -> None:
        """loaded() calls unload() when the block exits normally."""
        model = _ConcreteModel("default", Path("/x"))
        with model.loaded(MagicMock()):
            pass
        assert model.is_loaded is False

    def test_loaded_unloads_on_exception(self) -> None:
        """loaded() calls unload() even when an exception is raised."""
        model = _ConcreteModel("default", Path("/x"))
        with pytest.raises(ValueError, match="boom"):
            with model.loaded(MagicMock()):
                raise ValueError("boom")
        assert model.is_loaded is False

    def test_loaded_does_not_swallow_exception(self) -> None:
        """loaded() re-raises exceptions from the body."""
        model = _ConcreteModel("default", Path("/x"))
        with pytest.raises(RuntimeError, match="oops"):
            with model.loaded(MagicMock()):
                raise RuntimeError("oops")


@pytest.mark.unit
class TestBaseModelEnterExit:
    """Tests for __enter__ and __exit__."""

    def test_enter_returns_self(self) -> None:
        """__enter__ returns the model itself."""
        model = _ConcreteModel("default", Path("/x"))
        assert model.__enter__() is model

    def test_exit_calls_unload(self) -> None:
        """__exit__ calls unload()."""
        model = _ConcreteModel("default", Path("/x"))
        model.load(MagicMock())
        model.__exit__(None, None, None)
        assert model.is_loaded is False

    def test_with_block_unloads(self) -> None:
        """Using model as a with-block calls unload() on exit."""
        model = _ConcreteModel("default", Path("/x"))
        model.load(MagicMock())
        with model:
            assert model.is_loaded is True
        assert model.is_loaded is False


@pytest.mark.unit
class TestBaseModelDel:
    """Tests for __del__ GC safety net."""

    def test_del_unloads_loaded_model(self) -> None:
        """__del__ calls unload() if model is loaded."""
        model = _ConcreteModel("default", Path("/x"))
        model.load(MagicMock())
        assert model.is_loaded is True
        model.__del__()
        assert model.is_loaded is False

    def test_del_noop_when_not_loaded(self) -> None:
        """__del__ does nothing if model is not loaded."""
        model = _ConcreteModel("default", Path("/x"))
        model.__del__()  # Should not raise
        assert model.is_loaded is False

    def test_del_suppresses_unload_exceptions(self) -> None:
        """__del__ does not propagate exceptions from unload()."""

        class _FailingModel(BaseModel):
            def load(self, backend: object) -> None:
                """Load."""
                self._backend = backend  # type: ignore[assignment]

            def unload(self) -> None:
                """Always raises."""
                raise RuntimeError("unload failed")

        model = _FailingModel("default", Path("/x"))
        model.load(MagicMock())
        model.__del__()  # Should not raise
