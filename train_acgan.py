#!/usr/bin/env python3
import os
import json
import torch
import argparse
from tqdm import tqdm
from torch import nn

# Import your dataset class from your own code
from src.dataset_utils import get_dataset
# Import the CNN and ACGAN definitions from the ACGAN folder
from ACGAN.GAN.acgan_1 import ACGAN

# For the pretrained resnet18 model from the PyTorch model hub
import torchvision.models as models

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ACGAN using a custom dataset")
    parser.add_argument('--config', type=str, default='cfg/cifar_config_acgan.json',
                        help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = config['dataset']         # e.g. "mnist" or "cifar"
    experiment_name = config['experiment_name']

    # Create directory for saving the trained model
    save_path = os.path.join("models", experiment_name)
    os.makedirs(save_path, exist_ok=True)

    # Load your dataset using your own dataset class; fixed batch size=100.
    dataset = get_dataset(dataset_name)
    train_loader, test_loader = dataset.make_loaders(workers=4, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use the pretrained resnet18 from the PyTorch model hub.
    # Define the model architecture
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes

    # Load the model weights
    model.load_state_dict(torch.load('models/resnet18_cifar.pth'))
    model.to(device)
    model.eval()
    im_channel = 3  # resnet18 expects 3-channel images

    gan = ACGAN(
        in_dim=50,
        class_dim=10,
        g_filters=[384, 192, 96, 48, im_channel],
        g_strides=[1, 2, 2, 2],
        d_filters=[16, 32, 64, 128, 256, 512],
        d_strides=[2, 1, 2, 1, 2, 1],
        CNN=model
    )

    print("Starting ACGAN training ...")
    # The gan.train() method internally loops over epochs and batches.
    gan.train(train_loader, dataset.normalize, lr=config.get("lr", 0.0002),
              num_epochs=config.get("num_epochs", 100), test_loader=test_loader)

    # Save the generator and discriminator models after training.
    gan.save_model(save_path)
    print(f"Models saved to {save_path}")
