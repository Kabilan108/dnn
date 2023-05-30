"""
Functions & Classes for Model Training
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import pretrainedmodels as ptm
import torch

from tqdm.auto import tqdm
from glob import glob
import numpy as np
import re


def transform(config):
    """Tranforms for training, validation, and testing"""

    return transforms.Compose(
        [
            transforms.Resize((config["input-height"], config["input-width"])),
            transforms.ToTensor(),
            transforms.Normalize(config["input-mean"], config["input-std"]),
        ]
    )


def feature_dataloaders(config):
    """Create dataloaders for training, validation, and testing"""

    ds = {
        "train": datasets.ImageFolder(config["train-dir"], transform(config)),
        "test": datasets.ImageFolder(config["test-dir"], transform(config)),
        "val": datasets.ImageFolder(config["val-dir"], transform(config)),
    }

    return {
        "train": DataLoader(ds["train"], batch_size=config["batch-size"], shuffle=True),
        "test": DataLoader(ds["test"], batch_size=config["batch-size"], shuffle=True),
        "val": DataLoader(ds["val"], batch_size=config["batch-size"], shuffle=False),
    }


def load_ptm(config, replace_last=True):
    """Load a pretrained model and remove the last layer"""

    model = ptm.__dict__[config["model-name"]](num_classes=1000, pretrained="imagenet")
    if replace_last:
        model.last_linear = torch.nn.Identity()
    return model


def device():
    """Return the device to use for training"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureDataset(Dataset):
    """
    Custom dataset to load InceptionV3 features for fine-tuning
    """

    def __init__(self, feature_dir):
        self.files = sorted(glob(f"{feature_dir}/FL*.npy"), key=self._extract_idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label, feature = np.load(self.files[idx], allow_pickle=True)
        return torch.from_numpy(feature), torch.from_numpy(label)

    def _extract_idx(self, filename):
        """Extract batch index from filename"""
        match = re.search(r"(\d+)\.npy$", filename)
        match = int(match.group(1)) if match else -1
        if match == -1:
            raise ValueError(f"Invalid filename {filename}")
        return match


def accuracy(outputs, labels):
    """Compute accuracy of outputs compared to labels"""
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)


def pbar(dataloader, desc=None):
    """Create a tqdm progress bar"""
    return tqdm(dataloader, desc=desc, total=len(dataloader), leave=True, disable=False)
