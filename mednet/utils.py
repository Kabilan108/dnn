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

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_roc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend(loc="lower right")

    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.xlim([0.0, 1.05])
    plt.ylim((0.0, 1.05))

    return auc_roc
