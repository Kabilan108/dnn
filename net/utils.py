"""
Utility functions
"""

from sklearn.metrics import roc_auc_score, roc_curve, auc

import matplotlib.pyplot as plt
import glob
import os

import torchvision.transforms as transforms
import torch


def plot_roc(y_test, y_score, pos_label=0):
    """
    Plot a Receiver Operating Characteristic (ROC) curve and return the Area
    Under the Curve (AUC) score.

    Parameters
    ----------
    y_true: ndarray of shape (n_samples,)
        True binary labels.

    y_score: ndarray of shape (n_samples,)
        Target scores, (probability estimates of the positive class).

    pos_label: int or str, default=None
        The label of the positive class.
    """

    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=pos_label)
    auc_roc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, color="darkorange", label=f"ROC Curve (AUC = {auc_roc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.legend(loc="lower right", fancybox=False, frameon=False)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    # Major and minor grid lines
    ax.grid(which="major", color="#666666", linestyle="--", alpha=0.2)
    ax.minorticks_on()
    ax.grid(which="minor", color="#999999", linestyle="-.", alpha=0.1)
    ax.spines[["top", "right"]].set_visible(False)

    plt.xlim([0.0, 1.05])
    plt.ylim((0.0, 1.05))

    return auc_roc, (fig, ax)


def plot_history(history):
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
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(which="major", color="#666666", linestyle="--", alpha=0.2)
    axes[0].minorticks_on()
    axes[0].grid(which="minor", color="#999999", linestyle="-.", alpha=0.1)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(epochs, train_loss, "b", label="Training loss")
    axes[1].plot(epochs, val_loss, "r", label="Validation loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(which="major", color="#666666", linestyle="--", alpha=0.2)
    axes[1].minorticks_on()
    axes[1].grid(which="minor", color="#999999", linestyle="-.", alpha=0.1)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    return fig, axes
