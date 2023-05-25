"""
Utility functions
"""

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt
import numpy as np

import tarfile
import urllib
import glob
import sys
import os


def create_directory(path):
    """Create directory if it does not exist"""
    if not os.path.exists(path):
        os.makedirs(path)


def get_image_files(imagedir):
    """Return a sorted list of image files in a directory"""
    return sorted(
        [os.path.basename(fname) for fname in glob.glob(f"{imagedir}/*.jpeg")]
    )


def plot_roc(y_test, y_score, pos_label=0):
    """
    Plot a Receiver Operating Characteristic (ROC) curve and return the Area
    Under the Curve (AUC) score.

    Parameters
    ----------
    y_true: ndarray of shape (n_samples,)
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

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
