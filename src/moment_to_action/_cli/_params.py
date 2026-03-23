"""Custom click parameter types."""

import typing as t

import rich_click as click


class BasedIntParamType(click.ParamType):
    name = "integer"

    @t.override
    def convert(
        self,
        value: t.Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> int:
        """Convert the passed value to an integer."""
        if isinstance(value, int):
            return value

        try:
            return int(value, 0)
        except ValueError:
            self.fail(f"{value!r} is not a valid integer", param, ctx)


BASED_INT = BasedIntParamType()
