#!/usr/bin/env python3
import os
import json
import torch
import argparse
from tqdm import tqdm
from torch import nn

# Import your dataset class from your own code
from src.dataset_utils import get_dataset
from src.model_utils import resnet18_classifier

# Import the CNN and ACGAN definitions from the ACGAN folder
from ACGAN.GAN.acgan_1 import ACGAN
from ACGAN.GAN.acgan_res import ACGAN_Res

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

    dataset_name = 'cifar'        # e.g. "mnist" or "cifar"
    experiment_name = 'cifar_acgan'

    # Create directory for saving the trained model
    save_path = os.path.join("models", experiment_name)
    os.makedirs(save_path, exist_ok=True)

    # Load your dataset using your own dataset class; fixed batch size=100.
    dataset = get_dataset(dataset_name)
    train_loader, test_loader = dataset.make_loaders(workers=4, batch_size=256)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the CNN model that will be used as Discriminator in the ACGAN
    model = resnet18_classifier(device=device, dataset=dataset_name)
    im_channel = 3  # resnet18 expects 3-channel images
    in_dim = 500
    class_dim = 10

    if (dataset_name == 'mnist'):
         gan = ACGAN(
        in_dim=50,
        class_dim=10,
        g_filters=[384, 192, 96, 48, im_channel],
        g_strides=[1, 2, 2, 2],
        d_filters=[16, 32, 64, 128, 256, 512],
        d_strides=[2, 1, 2, 1, 2, 1],
        CNN=model
    )
    else:
        gan = ACGAN_Res(in_dim=in_dim, class_dim=class_dim, CNN=model)

   

    print("Starting ACGAN training ...")
    # The gan.train() method internally loops over epochs and batches.
    gan.train(train_loader, lr=0.0002, num_epochs=90)

    # Save the generator and discriminator models after training.
    gan.save_model(save_path)
    print(f"Models saved to {save_path}")
