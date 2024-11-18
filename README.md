# Detector Model for Adversarial Example Detection

## Overview

This repository contains the implementation of a detector model designed to identify adversarial examples in image datasets. The model employs multiple student networks to detect adversarial examples by analyzing the standard deviation and bias maps generated from the outputs of a teacher model and the student models.

## Requirements

- Python 3.x
- PyTorch
- robustness (a library for adversarial attacks)
- tensorboard
- cv2

Install the necessary packages using:
```bash
pip install torch torchvision tensorboard robustness
```

In order to be compatible with recent PyTorch versions, you may need to perform a modification to the modules in the robustness library. In the imagenet_models folders, in all .py files, `torchvision.models.utils` should be changed to `torch.hub`:

```python
from torch.hub import load_state_dict_from_url
```
