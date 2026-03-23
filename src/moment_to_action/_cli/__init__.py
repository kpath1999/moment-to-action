import logging
from pathlib import Path

import rich_click as click

from moment_to_action._logging import init_logging
from moment_to_action.utils.cli import GlobalData, ctx_get_seed, ctx_set_seed

from ._auto_group import auto_group
from ._params import BASED_INT


@auto_group(cmd_path=Path(__file__).parent / "commands")
@click.option(
    "-v",
    "--verbose",
    default=False,
    is_flag=True,
    help="Enable verbose logging.",
)
@click.option(
    "-s",
    "--seed",
    required=False,
    type=BASED_INT,
    default=None,
    help="Seed for random number generation.",
)
@click.pass_context
def cli(ctx: click.Context, *, verbose: bool, seed: int | None) -> None:
    """MTJ array simulation tool."""
    # Initialize logging
    init_logging(verbose=verbose)
    log = logging.getLogger("moment_to_action.cli")

    # Set global context data
    ctx.obj = GlobalData(log=log)

    # Set seed
    ctx_set_seed(ctx, seed)
    log.info("Running with seed %0#x", ctx_get_seed(ctx))
