"""
Use this script to train and evaluate models.
"""

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn, optim
import torch

from datetime import datetime
from rich import print
import click
import sys
import os

sys.path.append(".")
from net import train, utils, data


def check_data(config, download=True):
    """Check if data have been downloaded"""

    try:
        if not config["fine-tune"]:
            if not os.path.exists(config["features"]):
                raise FileNotFoundError
        else:
            for dir_key in ["train-dir", "val-dir", "test-dir"]:
                if not os.path.exists(config[dir_key]):
                    raise FileNotFoundError

        return True

    except FileNotFoundError:
        print("[red]Error:[/red] Data not found")
        print("Running [blue]python scripts/prepare_data.py[/blue]...")

    if download:
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

        # compute features if not fine-tuning
        if not config["fine-tune"]:
            train.compute_features(config)

        return True

    return False


@click.group()
def cli():
    """
    CLI to train and evaluate models
    """
    pass


@cli.command()
@click.option("-c", "--config", default="", help="Path to config file")
@click.option("-l", "--log", is_flag=True, help="Enable tensorboard logging")
@click.option("-v", "--verbose", is_flag=True, help="Print training progress")
@click.option(
    "--kw",
    default="",
    callback=utils.parse_kwargs,
    help="Comma separated list of kwargs to override config file",
)
def run(config, log, verbose, kw):
    """Run train-eval script"""

    # load config file
    try:
        config = utils.parse_config(config, kw)
    except FileNotFoundError:
        print("[red]Error:[/red] Config file not found")
        return

    # check if data is available
    if not check_data(config):
        return

    # fine-tuning vs feature extraction
    if config["fine-tune"]:
        if verbose:
            print("[green]Fine-tuning model[/green]")
        raise NotImplementedError
    else:
        if verbose:
            print("[green]Feature extraction - Training Classifier[/green]")

        # create datasets
        datasets = {
            split: train.FeatureDataset(f"{config['features']}/{split}")
            for split in ["train", "val", "test"]
        }

        # create data loaders
        dataloaders = {
            split: DataLoader(
                dataset,
                batch_size=config["batch-size"]["clf"],
                shuffle=split != "test",
            )
            for split, dataset in datasets.items()
        }

        # create the model
        model = train.FeatureExtractor(config)

    # move model to device
    model.to(train.device())

    # define optimizer and loss function
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable, lr=config["lr"])

    # define model name
    name = f"{config['name']}-{timestamp}"

    # initialize training and logging
    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    if log:
        writer = SummaryWriter(f"logs/{name}")
    else:
        writer = None
    EPOCH = 0

    # train the model
    best_model, history = train.train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config["epochs"],
        epoch_start=EPOCH,
        writer=writer,
        scheduler=None,
        is_inception=config["model-name"] == "inceptionv3",
    )
    EPOCH += config["epochs"]

    # evaluate the model
    accuracy, true_labels, pred_labels, pred_probs = train.evaluate_model(
        best_model, dataloaders["test"]
    )

    # figures
    cm, _ = utils.plot_confusion(true_labels, pred_labels, config)
    roc, _ = utils.plot_roc(true_labels, pred_probs, config)
    hist, _ = utils.plot_hist(history)

    # save figures
    if writer is not None:
        writer.add_figure("Confusion Matrix", cm, EPOCH)
        writer.add_figure("ROC Curve", roc, EPOCH)
        writer.add_figure("Loss History", hist, EPOCH)
    cm.savefig(f"figures/{name}-cm.png")
    roc.savefig(f"figures/{name}-roc.png")
    hist.savefig(f"figures/{name}-hist.png")

    # save model
    torch.save(best_model.state_dict(), f"models/{name}.pth")

    # write test results to file
    with open("data/performance_log.tsv", "a") as tsv:
        tsv.write(f"{name}\t\t{accuracy}\n")

    # print results
    if verbose:
        print(f"[green]Test Accuracy: {accuracy:.2f}[/green]")

    # close tensorboard writer
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    cli()
