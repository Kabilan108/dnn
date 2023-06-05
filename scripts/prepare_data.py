"""
Download and prepare dataset.

Run this script with the --help flag to see the options.
"""

from rich import print

import click
import sys

sys.path.append(".")
from net import train, data, utils


@click.command()
@click.option("--config", default="", help="Path to config file")
@click.option(
    "--kw",
    default="",
    callback=utils.parse_kwargs,
    help="Comma separated list of kwargs to override config file",
)
def main(config, kw):
    """CLI entry point"""

    # load config file
    try:
        config = utils.parse_config(config, kw)
    except FileNotFoundError:
        print("[red]Error:[/red] Config file not found")
        return

    # download and organize dataset
    if config["dataset"] == "xray":
        data.download_xray()
    elif config["dataset"] == "oct":
        data.download_oct()
    elif config["dataset"] == "mri":
        data.download_mri()
    else:
        print("[red]Error:[/red] Unknown dataset")
        return

    if not config["fine-tune"]:
        train.compute_features(config)

    return


if __name__ == "__main__":
    main()
