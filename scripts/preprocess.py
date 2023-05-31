"""
Preprocess Images

Run this script to run the images through the forward pass of the pretrained
model and save the output to disk.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

from tqdm.auto import tqdm
from rich import print

import sys

sys.path.append(".")
from net import train

import numpy as np
import shutil
import click
import json
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

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((config["input-height"], config["input-width"])),
            transforms.ToTensor(),
            transforms.Normalize(config["input-mean"], config["input-std"]),
        ]
    )

    # Define datasets
    data = {
        "train": datasets.ImageFolder(config["train-dir"], transform),
        "test": datasets.ImageFolder(config["test-dir"], transform),
        "val": datasets.ImageFolder(config["val-dir"], transform),
    }

    # Define dataloaders
    dataloaders = {
        "train": DataLoader(
            data["train"], batch_size=config["batch-size"], shuffle=True
        ),
        "test": DataLoader(data["test"], batch_size=config["batch-size"], shuffle=True),
        "val": DataLoader(data["val"], batch_size=config["batch-size"], shuffle=False),
    }

    # Load pretrained model
    model = train.load_ptm(config, replace_last=True, feature_extract=True)
    model = model.to(train.device())
    model.eval()

    # Create bottleneck directory
    shutil.rmtree(config["features"]["root"], ignore_errors=True)
    os.makedirs(config["features"]["root"], exist_ok=True)

    # Run images through model and save output to disk
    for split in ["train", "test", "val"]:
        for i, (inputs, labels) in enumerate(tqdm(dataloaders[split], desc=split)):
            # Move to GPU
            inputs = inputs.to(train.device())

            # Forward pass
            with torch.no_grad():
                features = model(inputs)
                features = features.cpu().detach().numpy()

            # Create directory for split
            root = config["features"][split]
            os.makedirs(root, exist_ok=True)

            # Save features to disk
            for j, feature in enumerate(features):
                label = labels[j].numpy()
                np.save(f"{root}/FL{i * len(inputs) + j}.npy", (label, feature))

    # Save class names and indices to disk
    with open(config["features"]["classes"], "w") as file:
        classes = data["train"].class_to_idx
        json.dump(classes, file)


if __name__ == "__main__":
    main()
