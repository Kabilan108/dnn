"""
Functions & Classes for Model Training
"""


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pretrainedmodels as models
import torch

from tqdm.auto import tqdm
from glob import glob
import numpy as np
import copy
import time
import re


def load_ptm(config, replace_last=True):
    """Load a pretrained model and remove the last layer"""

    model = models.__dict__[config["model-name"]](
        num_classes=1000, pretrained="imagenet"
    )
    if replace_last:
        model.last_linear = torch.nn.Identity()
    return model


def device():
    """Return the device to use for training"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pbar(dataloader, desc=None):
    """Create a tqdm progress bar"""

    return tqdm(
        dataloader,
        desc=desc,
        total=len(dataloader),
        leave=False,
        disable=False,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )


def inception_loss(outputs, aux_outputs, labels, criterion):
    """
    Loss function for training InceptionV3

    Inception is a special case during training because it produces two outputs:
    - outputs: the final layer
    - aux_outputs: an intermediate layer

    During training, we calculate the loss by taking a weighted sum of the
    final output and auxiliary output losses.
    During testing, validation, and inference, we only consider the final output.

    From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
    """

    loss1 = criterion(outputs, labels)
    loss2 = criterion(aux_outputs, labels)
    return loss1 + 0.4 * loss2


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs=25,
    epoch_start=0,
    writer=None,
    scheduler=None,
    is_inception=True,
):
    """
    Train a model
    """

    since = time.time()

    history = {k: {"acc": [], "loss": []} for k in ["train", "val"]}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epoch_start, epoch_start + num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluation mode

            # Track loss, TPs across batches
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data (batches)
            for inputs, labels in pbar(dataloaders[phase]):
                # Move tensors to the configured device
                inputs = inputs.to(device())
                labels = labels.to(device())

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Get model outputs and calculate loss
                    if is_inception and phase == "train":
                        outputs, aux_outputs = model(inputs)
                        loss = inception_loss(outputs, aux_outputs, labels, criterion)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # Determine predictions
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Update learning rate
            if phase == "train" and scheduler is not None:
                scheduler.step()

            # Calculate epoch statistics
            N = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / N
            epoch_acc = running_corrects.double() / N

            # Update history
            history[phase]["acc"].append(epoch_acc)
            history[phase]["loss"].append(epoch_loss)

            # Print epoch statistics
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Write to TensorBoard
            if writer is not None:
                writer.add_scalar(f"{phase}/loss", epoch_loss, epoch)
                writer.add_scalar(f"{phase}/acc", epoch_acc, epoch)

            # Deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history
