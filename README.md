# CIFAR-10 Image Classifier

A convolutional neural network built with PyTorch to classify images from the CIFAR-10 dataset into 10 categories: plane, car, bird, cat, deer, dog, frog, horse, ship, truck.

## Overview

This project implements a CNN from scratch using PyTorch, trained on the CIFAR-10 dataset (60,000 32x32 color images across 10 classes). It covers the full pipeline: data loading, preprocessing, model definition, training, and evaluation.

## Architecture

- 3 convolutional blocks (32 → 64 → 128 filters), each with BatchNorm, ReLU activation, and MaxPooling
- Dropout (0.25) for regularization
- 2 fully connected layers for final classification
- Trained using Adam optimizer and Cross-Entropy loss

## Requirements

- Python 3.12
- PyTorch
- torchvision

Install dependencies:
```bash
pip install torch torchvision
```

## How to Run this 

```bash
python cifar10classifier.py
```

The script will:
1. Automatically download the CIFAR-10 dataset (~170MB, first run only)
2. Train the model for 10 epochs
3. Evaluate accuracy on the test set
4. Save the trained model as `cifar10_cnn.pth`

## Results

- Test Accuracy: *(add your result here once training completes)*

## Notes

Training was tested on both CPU and Google Colab's free T4 GPU. GPU training is significantly faster and recommended for full runs.

## Author

Archit — B.Tech CSE (AI/ML)
