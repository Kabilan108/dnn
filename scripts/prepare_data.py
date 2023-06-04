"""
Download and prepare dataset.

Run this script with the --help flag to see the options.
"""

from torch.utils.data import DataLoader
from torchvision import datasets
import torch

from tqdm.auto import tqdm
from rich import print

import numpy as np
import shutil
import click
import json
import yaml
import sys
import os

sys.path.append(".")
from net import train, data, utils


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
@click.option("--config", default="", help="Path to config file")
@click.option(
    "--kw",
    default="",
    callback=parse_kwargs,
    help="Comma separated list of kwargs to override config file",
)
def main(config, kw):
    """CLI entry point"""

    # load config file
    try:
        with open(config, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("[red]Error:[/red] Config file not found")
        return

    # override config file with command line arguments
    for k, v in kw.items():
        config[k] = v

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
        # compose transforms
        transforms = utils.create_transforms(config["transforms"])

        # create data loaders
        dataloaders = {}
        for split in ["train", "val", "test"]:
            dataset = datasets.ImageFolder(config[f"{split}-dir"], transforms)
            dataloaders[split] = DataLoader(
                dataset, batch_size=config["batch-size"], shuffle=split != "test"
            )

        # Load pretrained model
        model = train.load_model(config, replace_last=True, feature_extract=True)
        model = model.to(train.device())
        model.eval()

        # Create directory for feature vectors
        shutil.rmtree(config["features"], ignore_errors=True)
        os.makedirs(config["features"], exist_ok=True)

        # Run images through model and save feature vectors
        for split in ["train", "val", "test"]:
            print(f"[bold blue]Extracting features for {split} set[/bold blue]")

            for i, (inputs, labels) in enumerate(tqdm(dataloaders[split], desc=split)):
                # move to gpu
                inputs = inputs.to(train.device())

                # forward pass
                with torch.no_grad():
                    features = model(inputs)

                # [ViT Only] extract embedding for CLS token
                if config["model-name"] == "ViT-base":
                    features = features.last_hidden_state[:, 0, :]

                # Convert to numpy
                features = features.cpu().detach().numpy()

                # Create directory for split
                root = f"{config['features']}/{split}"
                os.makedirs(root, exist_ok=True)

                # Save features to disk
                for j, feature in enumerate(features):
                    label = labels[j].numpy()
                    np.save(f"{root}/FL{i * len(inputs) + j}.npy", (label, feature))

        # save class names and indices to disk
        with open(f"{config['features']}/classes.json", "w") as file:
            classes = data["train"].class_to_idx
            json.dump(classes, file)

    return


if __name__ == "__main__":
    main()
