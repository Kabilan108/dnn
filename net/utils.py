"""
Utility functions
"""

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yaml
import json


def load_classes(config, ret="index"):
    """
    Load an ordered list of classes based on the dataset created when computing
    features
    """
    with open(f"{config['features']}/classes.json", "r") as f:
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

    # Adjust shape for two-class problem
    if true_labels.shape[1] == 1:
        true_labels = np.column_stack((1 - true_labels, true_labels))

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

    plt.tight_layout()

    return fig, ax


def plot_hist(history):
    """
    Plot training history
    """

    train_loss = history["train"]["loss"]
    val_loss = history["val"]["loss"]
    train_acc = history["train"]["acc"]
    val_acc = history["val"]["acc"]
    assert len(train_loss) == len(val_loss) == len(train_acc) == len(val_acc)

    epochs = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, "b", label="Training loss")
    axes[0].plot(epochs, val_loss, "r", label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Categorical Crossentropy Loss")
    axes[0].legend()
    axes[0].grid(which="major", color="#666666", linestyle="--", alpha=0.2)
    axes[0].minorticks_on()
    axes[0].grid(which="minor", color="#999999", linestyle="-.", alpha=0.1)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(epochs, train_acc, "b", label="Training accuracy")
    axes[1].plot(epochs, val_acc, "r", label="Validation accuracy")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(which="major", color="#666666", linestyle="--", alpha=0.2)
    axes[1].minorticks_on()
    axes[1].grid(which="minor", color="#999999", linestyle="-.", alpha=0.1)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    return fig, axes


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


def parse_config(config, kw=None):
    """Parse config file"""

    # load config file
    try:
        with open(config, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config} not found")

    # override config file with command line arguments
    if kw is not None:
        for k, v in kw.items():
            config[k] = v

    return config
