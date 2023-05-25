"""
Utility functions for Deep Learning
"""

from abc import ABC, abstractmethod
import torch


def fit(
    model,
    criterion,
    train_loader,
    val_loader,
    optimizer=None,
    scheduler=None,
    epochs=10,
    verbose=True,
):
    """
    Train and evaluate a PyTorch model

    Parameters
    ----------

    """

    # Keep track of metrics
    loss = {"train": [], "val": []}
    acc = {"train": [], "val": []}
