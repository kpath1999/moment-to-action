"""Verify QAIRT SDK installation command."""

from __future__ import annotations

import rich_click as click

from moment_to_action.qairt import QairtSDKManager
from moment_to_action.utils.cli import GlobalData, pass_global_data


@click.command()
@pass_global_data
def verify(data: GlobalData) -> None:
    """Verify the QAIRT SDK installation.

    Checks that the configured SDK path exists and all required system
    packages are installed. Exits 1 if any issues are found.
    """
    mgr = QairtSDKManager.from_app_config(data.config, data.path_manager)
    try:
        issues = mgr.verify(stream=True)
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e
    for issue in issues:
        data.log.warning(issue)
    if issues:
        raise SystemExit(1)
