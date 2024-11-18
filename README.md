# Detector Model for Adversarial Example Detection

## Overview

This repository contains the implementation of a detector model designed to identify adversarial examples in image datasets. The model employs multiple student networks to detect adversarial examples by analyzing the standard deviation and bias maps generated from the outputs of a teacher model and the student models.

## Requirements

- Python 3.x
- PyTorch
- robustness (a library for adversarial attacks)
- tensorboard
- OpenCV (cv2)

Install the necessary packages using:
```bash
pip install torch torchvision tensorboard robustness opencv-python
```

## Compatibility Note

To ensure compatibility with recent PyTorch versions, you may need to modify the robustness library. In the `imagenet_models` folder, update all `.py` files to replace `torchvision.models.utils` with `torch.hub`:

```python
from torch.hub import load_state_dict_from_url
```

## Project Structure

```
├── cfg/
│   └── ..
├── dataset_utils.py
├── detector.py
├── eval_utils.py
├── model_utils.py
├── models/
│   └── ...
├── pretraining.py
├── README.md
├── requirements.txt
├── results/
│   └── ...
├── run.sh
├── runs/
│   └── ...
├── test.py
└── train.py
```

## Usage

### Training

To train the detector model, run the following command:

```bash
python train.py --config cfg/mnist_config.json
```

### Testing

To evaluate the detector model, run the following command:

```bash
python test.py --config cfg/mnist_config.json
```

You can replace `cfg/mnist_config.json` with the path to any other configuration file as needed.

### Running All Experiments

To run all experiments sequentially, use the provided shell script:

```bash
sh run.sh
```

## Configuration

The configuration files are located in the `cfg` directory. Each configuration file specifies the parameters for training and testing, such as the dataset, patch size, number of students, and attack parameters.

Example configuration (`cfg/mnist_config.json`):

```json
{
    "experiment_name": "mnist_exp000",
    "dataset": "mnist",
    "patch_size": 17,
    "num_students": 10,
    "train": {
        "steps": 100000
    },
    "test": {
        "attacker": {
            "epsilon": 0.1,
            "constraint": "inf",
            "targeted": false,
            "iterations": 100,
            "step_size": 0.01 
        },
        "samples": 5000,
        "save": true
    }
}
```

## Results

The results of the experiments are saved in the `results` directory. Each experiment has its own subdirectory containing the saved models and evaluation metrics.
