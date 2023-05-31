# Evaluating Vision Transformers for Medical Image Classification

## Abstract

## Quick Start

### Set up the environment

1. Install [anaconda](https://docs.anaconda.com/anaconda/install/) or  or [miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Clone the repo:

```bash
git clone git@github.com:kabilan108/ViT-Med.git
cd ViT-Med
```

3. Create a conda environment:

```bash
conda env create -f environment.yml
conda activate vit-med
```

4. Download the OCT and Chest X-Ray datasets:

```bash
./scripts/download_data.sh
```

### Training & Validation

### Evaluation

### Inference

## Background

### Datasets

All datasets utilized for this project were obtained from Kaggle and were
sourced from one or more academic publications. The following datasets were
used:

- [Retinal Optical Coherence Tomography (OCT)](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
  - paultimothymooney/kermany2018
  - Based on Kermany et. al. (2018)
  - 84,495 images across 4 classes (CNV, DME, DRUSEN, NORMAL)
    - Train/Test/Val split?
- [Pediatric Chest X-Rays](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray)
  - paultimothymooney/chest-xray-pneumonia
  - Based on Kermany et. al. (2018)
  - 5856 images across 2 classes (PNEUMONIA, NORMAL)
    - Train/Test/Val split?
- [Brain Tumor MRIs](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
  - masoudnickparvar/brain-tumor-mri-dataset
  - Includes data from
    - Cheng, Jun (2017). brain tumor dataset. figshare. Dataset. https://doi.org/10.6084/m9.figshare.1512427.v5
    - sartajbhuvaji/brain-tumor-classification-mri
    - brain-tumor-detection?select=Br35H-Mask-RCNN
  - 7022 images across 4 classes (MENINGIOMA, GLIOMA, PITUITARY, NORMAL)
    - Train/Test/Val split?
