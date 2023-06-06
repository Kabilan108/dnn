# Evaluation of State-of-the-Art Models for Medical Image Classification

## Problem Statement

Medical imaging serves as a critical diagnostic tool, yet its use is often hampered by challenges related to image quality, reliability and interoperability. These challenges stem from the variability of diagnostic images, leading to uncertainty and potential inaccurate diagnoses, subsequently guiding clinicians towards improper treatment strategies. Deep Neural Networks (DNNs) have emerged as a promising solution to these issues. Specifically, Convolutional Neural Networks (CNNs) have shown significant potential in the identification of diseases in medical images. For instance, research by Kermany et. al. demonstrated the efficacy of a fine-tuned CNN in diagnosing a variety of retinal diseases in Optical Coherence Tomography (OCT) scans, as well as pneumonia in pediatric chest radiographs [1]. In 2017, Esteva et. al. fine-tuned a similar CNN to diagnose over 20 unique skin conditions, and achieved performance on par with a team of board-certified dermatologists [2].

Nonetheless, it's crucial to understand that computational classification is not limited to CNNs. In the wake of large language models' emergence, the transformer architecture has gained significant traction across numerous tasks in natural language processing​ [3]​. The transformer's application has extended to computer vision, marked by the development of various Vision Transformers (ViTs) that have proven to be highly effective in object segmentation tasks [4]​. However, a comprehensive investigation into the comparative performance of these models in medical image classification tasks remains to be conducted.

This research aims to bridge the gap in medical imaging diagnostics by evaluating the performance of state-of-the-art CNN models, the Inception V3 architecture used by Kermany et. al., with that of the more recent ViTs across a variety of medical imaging modalities including OCT scans, radiographs (X-rays), and Magnetic Resonance Imaging (MRIs).

By undertaking this research, we aim to contribute to the refinement of current diagnostic capabilities and potentially inspire further exploration of alternative neural network methodologies in medical imaging. Our hope is that the results could lead to a more robust and standardised approach to diagnostic care, thus reducing the uncertainty faced by many clinicians when using medical imaging for diagnostics.


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

4. Train a model:

```bash
python scripts/train_eval -lv -c config/<YOUR_CONFIG>.yaml
```

### Running in Colab

* Click the "Open in Colab" button in the [VSCode-Server](notebooks/VSCode-Server.ipynb) notebook
* Run the top cell to install dependencies, mount your Google Drive, and clone the repo
* This will also start an SSH server on the VM
  * Copy the connect to the session via local VSCode
