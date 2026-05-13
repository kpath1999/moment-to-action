import logging
from pathlib import Path

import rich_click as click

from moment_to_action._logging import init_logging
from moment_to_action.config import load_config
from moment_to_action.paths import PathManager
from moment_to_action.utils.cli import GlobalData, ctx_get_seed, ctx_set_seed

from ._auto_group import auto_group
from ._params import BASED_INT


@auto_group(cmd_path=Path(__file__).parent / "commands")
@click.option(
    "-v",
    "--verbose",
    default=False,
    is_flag=True,
    help="Enable verbose logging (overrides config log_level).",
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
    # Load config first so log_level is available
    path_manager = PathManager()
    config = load_config(path_manager.app_config_file)

    # Init logging
    init_logging(log_level="DEBUG" if verbose else config.log_level)
    log = logging.getLogger("moment_to_action.cli")

    # Build global data object
    ctx.obj = GlobalData(log=log, path_manager=path_manager, config=config)

    # Set seed
    ctx_set_seed(ctx, seed)
    log.info("Running with seed %0#x", ctx_get_seed(ctx))
