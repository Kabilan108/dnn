"""
Functions for loading and organizing datasets from kaggle
"""

import subprocess


def download_xray():
    """
    Download the paultimothymooney/chest-xray-pneumonia dataset from kaggle
    """

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

    return


def download_oct():
    """
    Download the paultimothymooney/kermany2018 dataset from kaggle
    """

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

    return


def download_mri():
    """
    Download MRI dataset
    """

    raise NotImplementedError
