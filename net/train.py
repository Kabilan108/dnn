"""
Functions & Classes for Model Training
"""


from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import functional as F
import torch

from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from glob import glob
import copy
import json
import time
import re


def load_ptm(config, feature_extract=True, replace_last=False):
    """Load a pretrained model and remove the last layer"""

    if config["model-name"] == "inceptionv3":
        model = models.inception_v3(weights="IMAGENET1K_V1")
    else:
        raise NotImplementedError

    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    if replace_last:
        if config["model-name"] == "inceptionv3":
            model.AuxLogits.fc = torch.nn.Identity()
            model.fc = torch.nn.Identity()
        else:
            raise NotImplementedError

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
        print(f"Epoch {epoch + 1}/{num_epochs + epoch_start}")
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
            print(f"{phase:6} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

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


def evaluate_model(model, test_loader):
    """Evaluate model performance"""

    model.eval()

    # store true and predicted labels, and probabilities of each class
    true_labs, pred_labs, pred_probs = [], [], []

    # counters
    total, corrects = 0, 0

    for inputs, labels in test_loader:
        # move to device
        inputs = inputs.to(device())
        labels = labels.to(device())

        # forward pass
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # compute probabilities
            probs = F.softmax(outputs, dim=1)

            # Move data to cpu
            labels = labels.cpu().numpy()
            preds = preds.cpu().numpy()
            probs = probs.cpu().numpy()

            # Update lists
            true_labs.extend(labels)
            pred_labs.extend(preds)
            pred_probs.extend(probs)

            # Update counters
            total += len(labels)
            corrects += np.sum(labels == preds)

    # Calculate accuracy
    acc = corrects / total * 100

    # Convert to numpy arrays
    true_labs = np.array(true_labs)
    pred_labs = np.array(pred_labs)
    pred_probs = np.array(pred_probs)

    return acc, true_labs, pred_labs, pred_probs


def load_classes(config, ret="index"):
    """
    Load an ordered list of classes based on the dataset created when computing
    features
    """
    with open(config["features"]["classes"], "r") as f:
        ctoi = json.load(f)

    if ret == "label":
        return sorted(ctoi, key=ctoi.get)
    elif ret == "index":
        return sorted(ctoi.values())
    elif ret == "dict":
        return ctoi
    else:
        raise ValueError("Invalid return type")


def plot_confusion(true_labels, pred_labels, config):
    """Plot a confusion matrix"""
    classes = load_classes(config, ret="label")
    cm = pd.DataFrame(
        confusion_matrix(true_labels, pred_labels), index=classes, columns=classes
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, cmap="Reds", fmt="d", cbar=False, ax=ax)
    ax.set_ylabel("True Labels", fontsize=11)
    ax.set_xlabel("Predicted Labels", fontsize=11)

    return fig, ax


def plot_roc(true_labels, pred_probs, config):
    """Plot ROC curve and compute AUC"""

    # Convert labels to binary format for one-vs-rest ROC curve
    classes = load_classes(config, ret="index")
    labels = load_classes(config, ret="label")
    true_labels = label_binarize(true_labels, classes=classes)

    # Initialize variables
    fpr, tpr, roc_auc = {}, {}, {}

    # Compute ROC curve and ROC area for each class
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    fig, ax = plt.subplots(figsize=(4, 4))

    for i, label in enumerate(labels):
        ax.plot(fpr[i], tpr[i], lw=1.5, label=f"{label} (AUC = {roc_auc[i]:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim([-0.05, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(which="major", color="#666666", linestyle="--", alpha=0.2)
    ax.minorticks_on()
    ax.grid(which="minor", color="#999999", linestyle="-.", alpha=0.1)

    return fig, ax
