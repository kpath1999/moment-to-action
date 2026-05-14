"""Utilites for working with pydantic schemas."""

from __future__ import annotations

import typing as t

from pydantic import BaseModel

_T = t.TypeVar("_T", bound=BaseModel)


def update_frozen(model: _T, **updates: t.Any) -> _T:
    """Update a pydantic model, returning a copy.

    VERY useful for frozen models.

    Args:
        model: Model to update.
        updates: key=value pairs to update.

    Returns:
        The updated model.

    Raises:
        ValidationError: invalid value in updates.
    """
    # Create new data; updates win here
    # https://docs.python.org/3/library/stdtypes.html#dict.values
    data = model.model_dump() | updates

    # Validate new data
    return type(model).model_validate(data)
