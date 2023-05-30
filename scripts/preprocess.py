"""
Preprocess Images

Run this script to run the images through the forward pass of the pretrained
model and save the output to disk.
"""

from tqdm.auto import tqdm
from rich import print

import sys

sys.path.append(".")
from net import train

import numpy as np
import shutil
import click
import yaml
import os


def parse_kwargs(ctx, param, value):
    """Parse kwargs from command line"""
    if not value:
        return {}

    kwargs = {}
    for kv in value.split(","):
        k, v = kv.split("=")

        try:
            kwargs[k] = int(v)
        except ValueError:
            try:
                kwargs[k] = float(v)
            except ValueError:
                kwargs[k] = v

    return kwargs


@click.command()
@click.option("--config", default="config.yaml", help="Path to config file")
@click.option(
    "--kw",
    default="",
    callback=parse_kwargs,
    help="Comma separated list of kwargs to override config file",
)
def main(config, kw):
    """Script entry point"""

    # Load config file
    try:
        with open(config, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("[red]Error:[/red] Config file not found")
        return

    # Override config file with kwargs
    for k, v in kw.items():
        config[k] = v

    print("[green]Config:[/green]")
    print(config)

    # Create dataloaders
    dl = train.feature_dataloaders(config)

    # Load pretrained model
    model = train.load_ptm(config)
    model = model.to(train.device())

    # Create bottleneck directory
    shutil.rmtree(config["features"]["root"], ignore_errors=True)
    os.makedirs(config["features"]["root"], exist_ok=True)

    # Run images through model and save output to disk
    for split in ["train", "test", "val"]:
        for i, (inputs, labels) in enumerate(tqdm(dl[split], desc=split)):
            inputs = inputs.to(train.device())
            features, _ = model(inputs)
            features = features.cpu().detach().numpy()

            root = config["features"][split]
            os.makedirs(root, exist_ok=True)

            for j, feature in enumerate(features):
                label = labels[j].numpy()
                np.save(f"{root}/FL{i * len(inputs) + j}.npy", (label, feature))


if __name__ == "__main__":
    main()
