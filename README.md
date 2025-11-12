# Uncertainty-Guided Physics-Informed Image Registration for Brain Tumor MRI with Missing Correspondences
This is the official PyTorch implementation of "Uncertainty-Guided Physics-Informed Image Registration for Brain Tumor MRI with Missing Correspondences", written by Y. L. Lyndon Chan and Albert C. S. Chung.

## Prerequisites
- `Python 3.8+`
- `PyTorch 2.0.0+`
- `NumPy`
- `NiBabel`
- `Scipy`

This code has been tested with `Pytorch 2.8.0` and NVIDIA L40S GPU.

## Inference

Inference for UPIR:
```
python evaluate.py
```

## Train your own model
Step 1: Download the BraTS-Reg dataset from https://zenodo.org/records/14642405.

Step 2: Define and split the dataset into training and validation set, or run the five-fold cross validation directly as in train.py.

Step 3: `python train.py` to train the UPIR model.


###### Keywords
Keywords: Deformable image registration, Missing anatomical correspondences, Biomechanical constraints