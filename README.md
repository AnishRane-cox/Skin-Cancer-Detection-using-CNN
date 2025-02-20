# Skin Cancer Classification using Deep Learning

> A deep learning-based approach for skin cancer detection using the ISIC dataset and data augmentation techniques to improve model performance.

## Table of Contents
* [General Information](#general-information)
* [Dataset and Preprocessing](#dataset-and-preprocessing)
* [Data Augmentation](#data-augmentation)
* [Model Architecture](#model-architecture)
* [Training and Results](#training-and-results)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Information
- This project aims to classify different types of skin cancer using deep learning.
- It addresses the issue of class imbalance and improves model generalization using data augmentation.
- The dataset used is the ISIC (International Skin Imaging Collaboration) dataset.
- The model is trained using TensorFlow and Keras with techniques like dropout and data augmentation.

## Dataset and Preprocessing
- The dataset consists of multiple skin lesion classes, and initially, there was an imbalance among classes.
- To resolve class imbalance, the `Augmentor` library was used to generate synthetic images.
- The dataset was split into training and validation sets, ensuring one-hot encoding for multi-class classification.

## Data Augmentation
- Used `Augmentor` to generate 500 additional samples per class to balance the dataset.
- Applied transformations such as:
  - Rotation (±10 degrees)
  - Flipping
  - Scaling
- Ensured that augmented data was properly integrated into the TensorFlow dataset pipeline.

## Model Architecture
- CNN-based model with the following layers:
  - Convolutional layers with ReLU activation
  - Batch normalization
  - Dropout layers to prevent overfitting
  - Fully connected layers for classification
- Used categorical cross-entropy as the loss function due to multi-class classification.

## Training and Results
- **Epochs:** 20
- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy
- **Training Accuracy:** Improved from ~20% to ~55% after augmentation and dropout adjustments.
- **Validation Accuracy:** Increased but still required fine-tuning to improve generalization.
- Checked for signs of underfitting/overfitting and made necessary adjustments.

## Technologies Used
- Python 3.x
- TensorFlow/Keras
- Augmentor
- NumPy
- Pandas
- Matplotlib
- Glob

## Conclusions
- Data augmentation significantly improved class balance, leading to better generalization.
- Dropout layers helped prevent overfitting, stabilizing the validation accuracy.
- Further improvements can be made using transfer learning with pre-trained models (e.g., ResNet, EfficientNet).

## Acknowledgements
- Inspired by the ISIC Skin Cancer Challenge.
- References from TensorFlow and Keras documentation.
- Augmentor library documentation for handling class imbalance.

## Contact
Created by [@AnishRane-cox] - feel free to reach out!
