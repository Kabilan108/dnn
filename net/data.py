"""
Functions for loading and organizing datasets from kaggle
"""

import subprocess
import shutil
import random
import os

random.seed(42)

VAL_SPLIT = 0.1


def download_xray():
    """
    Download the paultimothymooney/chest-xray-pneumonia dataset from kaggle
    """

    try:
        subprocess.run(
            """
            kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
            unzip -q "chest-xray-pneumonia.zip" -d "data"
            rm -r "chest-xray-pneumonia.zip" "data/chest_xray/__MACOSX"
            rm -r "data/chest_xray/chest_xray"
        """,
            shell=True,
            check=True,
        )
    except KeyboardInterrupt:
        print("Aborted")
        return

    return


def download_oct():
    """
    Download the paultimothymooney/kermany2018 dataset from kaggle
    """

    try:
        subprocess.run(
            """
            kaggle datasets download -d paultimothymooney/kermany2018
            unzip -q "kermany2018.zip" -d "data/oct"
            mv data/oct/OCT2017\ /t* data/oct && mv data/oct/OCT2017\ /v* data/oct
            rmdir data/oct/OCT2017\ /
            rm -r "data/oct/oct2017" "kermany2018.zip"
        """,
            shell=True,
            check=True,
        )
    except KeyboardInterrupt:
        print("Aborted")
        return

    return


def download_mri():
    """
    Download MRI dataset
    """

    try:
        subprocess.run(
            """
            kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
            unzip -q "brain-tumor-mri-dataset.zip" -d "data/mri"
            mv data/mri/Testing data/mri/test
            mv data/mri/Training data/mri/train
            mkdir data/mri/val
        """,
            shell=True,
            check=True,
        )
    except KeyboardInterrupt:
        print("Aborted")
        return

    # path to test dir
    test_dir = "data/mri/test"
    val_dir = "data/mri/val"

    # ensure validation directory exists and is empty
    if os.path.exists(val_dir):
        subprocess.run(f"rm -r {val_dir}", shell=True, check=True)
    os.mkdir(val_dir)

    # iterate over classes
    for class_name in os.listdir(test_dir):
        class_test_dir = os.path.join(test_dir, class_name)
        class_val_dir = os.path.join(val_dir, class_name)

        # ensure class validation directory exists
        if not os.path.exists(class_val_dir):
            os.mkdir(class_val_dir)

        # get list of (non-directory) files in class test directory
        files = [
            f
            for f in os.listdir(class_test_dir)
            if os.path.isfile(os.path.join(class_test_dir, f))
        ]

        # select random subset of files to use for validation
        N = int(len(files) * VAL_SPLIT)
        val_files = random.sample(files, N)

        # Move the selected files to the validation directory
        for file in val_files:
            shutil.move(os.path.join(class_test_dir, file), class_val_dir)

    return
