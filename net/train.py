"""
Functions & Classes for Model Training
"""

from torchvision import models, datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from transformers import ViTModel
from torch import nn
import torch

from tqdm.auto import tqdm
from glob import glob
import numpy as np
import shutil
import copy
import json
import time
import re
import os


def load_ptm(config, feature_extract=True, replace_last=False):
    """Load a pretrained model and remove the last layer"""

    if config["model-name"] == "inceptionv3":
        model = models.inception_v3(weights="IMAGENET1K_V1")
    elif config["model-name"] == "ViT-base":
        model = ViTModel.from_pretrained(config["checkpoint"], add_pooling_layer=False)
    else:
        raise NotImplementedError

    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    if replace_last:
        if config["model-name"] == "inceptionv3":
            model.AuxLogits.fc = torch.nn.Identity()
            model.fc = torch.nn.Identity()
        elif config["model-name"] == "ViT-base":
            pass
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


def create_transforms(transforms_list):
    """Compose transformations for data augmentation"""

    transform = []
    for item in transforms_list:
        name, args = item
        tclass = getattr(transforms, name)
        transform.append(tclass(**args))

    return transforms.Compose(transform)


def compute_features(config):
    """Compute features from a feature extractor"""

    # compose transforms
    transforms = create_transforms(config["transforms"])

    # create data loaders
    data, dataloaders = {}, {}
    for split in ["train", "val", "test"]:
        data[split] = datasets.ImageFolder(config[f"{split}-dir"], transforms)
        dataloaders[split] = DataLoader(
            data[split],
            batch_size=config["batch-size"]["feature-extraction"],
            shuffle=split != "test",
        )

    # Load pretrained model
    model = load_ptm(config, replace_last=True, feature_extract=True)
    model = model.to(device())
    model.eval()

    # Create directory for feature vectors
    shutil.rmtree(config["features"], ignore_errors=True)
    os.makedirs(config["features"], exist_ok=True)

    # Run images through model and save feature vectors
    for split in ["train", "val", "test"]:
        print(f"[bold blue]Extracting features for {split} set[/bold blue]")

        for i, (inputs, labels) in enumerate(tqdm(dataloaders[split], desc=split)):
            # move to gpu
            inputs = inputs.to(device())

            # forward pass
            with torch.no_grad():
                features = model(inputs)

            # [ViT Only] extract embedding for CLS token
            if config["model-name"] == "ViT-base":
                features = features.last_hidden_state[:, 0, :]

            # Convert to numpy
            features = features.cpu().detach().numpy()

            # Create directory for split
            root = f"{config['features']}/{split}"
            os.makedirs(root, exist_ok=True)

            # Save features to disk
            for j, feature in enumerate(features):
                label = labels[j].numpy()
                np.save(f"{root}/FL{i * len(inputs) + j}.npy", (label, feature))

    # save class names and indices to disk
    with open(f"{config['features']}/classes.json", "w") as file:
        classes = data["train"].class_to_idx
        json.dump(classes, file)

    return True


class FeatureDataset(Dataset):
    """Custom dataset for using pre-trained feature extractors"""

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


class FeatureExtractor(nn.Module):
    """Classifier for pre-trained feature extractors"""

    def __init__(self, config):
        super(FeatureExtractor, self).__init__()

        self.clf = nn.Linear(config["embedding-dim"], config["num-classes"])

    def forward(self, x):
        x = self.clf(x)
        return x


class FineTuned(nn.Module):
    """Fine-tuned model"""

    def __init__(self, config):
        super(FineTuned, self).__init__()

        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
