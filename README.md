# Detector Model for Adversarial Example Detection

## Overview

This repository contains the implementation of a detector model designed to identify adversarial examples in image datasets. The model employs multiple student networks to detect adversarial examples by analyzing the standard deviation and bias maps generated from the outputs of a teacher model and the student models.

## Requirements

- Python 3.x
- PyTorch
- robustness (a library for adversarial attacks)
- tensorboard

Install the necessary packages using:
```bash
pip install torch torchvision tensorboard robustness
```

## Detector Class

The `Detector` class is the core of this implementation. It comprises several key components:

### Attributes

- **device**: The device used for training (CPU or GPU).
- **dataset**: The dataset object (CIFAR-10 in this case).
- **patch_size**: The size of the patches extracted from images.
- **teacher**: The teacher model used for feature extraction.
- **students**: A list of student models trained to mimic the teacher model.
- **criterion**: The loss function (Mean Squared Error).
- **optimizer**: The optimizer (Adam) for training the student models.
- **attacker**: The attacker object used to generate adversarial examples.
- **attack_kwargs**: The keyword arguments for configuring the attacker.
- **writer**: The TensorBoard writer for logging.

### Methods

- **__init__**: Initializes the detector with the given number of student models, attack configuration, patch size, and device.
- **evaluate**: Evaluates the model on the test dataset and logs statistics to TensorBoard.
- **train_patch**: Trains the student models on a batch of patches.
- **train_model**: Trains the detector model for a specified number of epochs.
- **train**: Sets the training mode for all student models.
- **eval**: Sets the evaluation mode for all student models.
- **forward**: Performs a forward pass through the model to compute the standard deviation and bias maps.

## Usage

### Command-Line Arguments

The script accepts several command-line arguments to configure the adversarial attack:
- `--epsilon`: Epsilon constraint for the attack (default: 0.1).
- `--linf`: Whether to use L-inf norm for the attack (default: False).
- `--targeted`: Whether the attack is targeted (default: False).
- `--iterations`: Number of PGD steps (default: 100).

### Example Usage

```bash
python detector.py --epsilon 0.1 --linf False --targeted False --iterations 100
```

### Training the Model

The `Detector` class can be instantiated and trained using the following code:

```python
from robustness.datasets import CIFAR

# Configuration for the adversarial attack
attack_kwargs = {
    'constraint': 'inf',  # L-inf PGD 
    'eps': 0.1,  # Epsilon constraint (L-inf norm)
    'step_size': 0.01,  # Learning rate for PGD
    'iterations': 100,  # Number of PGD steps
    'targeted': False,  # Targeted attack
    'custom_loss': None  # Use default cross-entropy loss
}

# Initialize the detector with 10 student models and patch size of 9
detector = Detector(10, attack_kwargs, patch_size=9)

# Create data loaders
train_loader, test_loader = detector.dataset.make_loaders(workers=4, batch_size=1)

# Train the detector model
detector.train_model(train_loader, test_loader, num_epochs=10, interval=10000)
```

## Conclusion

This implementation provides a robust framework for detecting adversarial examples using multiple student models. By leveraging the TensorBoard for logging, it offers insights into the model's performance and the characteristics of adversarial examples.