
This repository implements a Vision Transformer for classifying images using triplet loss, focusing on detecting Waldo from the "Where's Waldo" series.



## Introduction

*Where's Waldo?* is a children's book series and game. Readers search for Waldo, a tall man in a red striped shirt, red beanie, and glasses, hidden among busy scenes. The goal is to find Waldo as quickly as possible. For this project, we obtained a dataset from the Hey-Waldo repository on GitHub.

## Code Explanation

### Imports and Setup
- The necessary libraries for deep learning (PyTorch), image processing, and data manipulation are imported.
- The random seeds are set for reproducibility.
- The device (CPU or GPU) is selected for training.

### Data Loading
- A custom `WaldoPatchDataset` class is defined to load image patches from directories, resize them, and convert them to tensors.
- The dataset is split into training, validation, and test sets, and the distribution of labels is printed.

### Data Processing
- A function `compute_mean_std` calculates the mean and standard deviation of the RGB channels in the training dataset for normalization purposes.

### Augmented and Triplet Datasets
- An `AugmentedDataset` class is defined to handle data augmentation selectively applied to positive samples to address class imbalance.
- A `TripletWaldoDataset` class is created to prepare triplets (anchor, positive, and negative samples) for training with Triplet Loss.
- Normalization and augmentation transforms are defined and applied to the datasets.
- Data loaders for the triplet datasets are created.

### Vision Transformer Implementation
- The Vision Transformer model is implemented with classes for patchification (`Patchify`), feedforward neural network (`FeedForward`), multi-head attention (`MultiHeadAttention`), and the main transformer (`Transformer`).
- The `ViT` class integrates these components and includes a classifier.

### Triplet Loss Implementation
- The `TripletLoss` class defines the triplet loss function used to train the model by minimizing the distance between similar images and maximizing the distance between dissimilar images.

### Training and Evaluation
- Functions `train_triplet_epoch` and `evaluate_triplet` are defined to handle training and evaluation of the model using the triplet loss.
- The model is trained for a specified number of epochs, with loss and distance metrics being recorded.
- The best model state based on validation loss is saved.

### Results Visualization and Model Saving
- Training and test distances, as well as loss values, are plotted and saved as images.
- The final model state is saved to a file for future use.

## Key Components Explained

### Custom Dataset and DataLoader
- `WaldoPatchDataset` loads images and labels, resizes them, and converts them to tensors.
- `AugmentedDataset` and `TripletWaldoDataset` handle data augmentation and creation of triplets for training with triplet loss.

### Normalization and Augmentation
- Data normalization ensures consistent input values across the dataset.
- Augmentation helps to mitigate class imbalance by creating variations of positive samples.

### Vision Transformer (ViT)
- `Patchify`: Divides images into patches and projects them into an embedding space.
- `FeedForward` and `MultiHeadAttention`: Core components of the transformer model handling transformations and attention mechanisms.
- `Transformer`: Sequentially applies multiple layers of attention and feedforward networks.
- `ViT`: Combines all components and includes a classifier for final output.

### Triplet Loss
- Used to train the model by ensuring that the distance between an anchor and positive sample is less than the distance between an anchor and negative sample.

### Training and Evaluation
- Training involves minimizing the triplet loss over epochs, adjusting the model parameters.
- Evaluation checks the model's performance on validation and test sets, focusing on the mean distances between positive and negative samples.

### Visualization
- Results are visualized to understand the model's learning progress and to ensure it is not overfitting.


