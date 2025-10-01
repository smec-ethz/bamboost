from typing import Optional

import typer

from bamboost.cli.common import console

app_config = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Print or edit the configuration.",
)


@app_config.command()
def print(
    dir: Optional[str] = typer.Option(
        None, "--dir", "-d", help="Directory to show the config for."
    ),
):
    """Show the active configuration."""
    if dir:
        from bamboost._config import _Config

        config = _Config(dir)
    else:
        from bamboost import config

    console.print(config)


@app_config.command()
def show():
    """Open the global config file."""
    import os

    from bamboost._config import CONFIG_FILE

    os.system(f"${{EDITOR:-vim}} {CONFIG_FILE}")
