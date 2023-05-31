"""
Utility functions
"""

from sklearn.metrics import roc_auc_score, roc_curve, auc

import matplotlib.pyplot as plt
import glob
import os

import torchvision.transforms as transforms
import torch


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


def save_model(model, path):
    """Save trained model (.pth) to path"""
    torch.save(model.state_dict(), path)


def create_image_lists(image_dir):
    """
    Take in image directory, return a dictionary containing images split into
    training, testing and validation splits
    """
    image_lists = {}
    label_dirs = glob.glob(os.path.join(image_dir, "*"))
    for label_dir in label_dirs:
        label_name = os.path.basename(label_dir)
        image_lists[label_name] = {
            "training": get_image_files(os.path.join(label_dir, "train")),
            "testing": get_image_files(os.path.join(label_dir, "test")),
            "validation": get_image_files(os.path.join(label_dir, "validation")),
        }
    return image_lists


def get_image_path(image_lists, label_name, index, image_dir, category):
    image_file = image_lists[label_name][category][index]
    return os.path.join(image_dir, label_name, category, image_file)


# Return path to bottleneck for a given label and a given index
def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category):
    bottleneck_file = (
        os.path.splitext(
            get_image_path(image_lists, label_name, index, bottleneck_dir, category)
        )[0]
        + ".pt"
    )
    return bottleneck_file


def imshow(image, title=None):
    """Visualize Image tensor"""
    global config
    image = image.numpy().transpose((1, 2, 0))
    image = config["input-std"] * image + config["input-mean"]
    image = np.clip(image, 0, 1)

    plt.figure()
    plt.imshow(image)
    if title is not None:
        plt.title(title)


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
    plt.show()
