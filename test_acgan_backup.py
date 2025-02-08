#!/usr/bin/env python3
import os
import json
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
from torch import nn

# Import your dataset class from your code
from src.dataset_utils import get_dataset
# Import ACGAN modules from the ACGAN folder
from ACGAN.GAN.acgan_1 import ACGAN

# Use the pretrained ResNet18 from torchvision
import torchvision.models as models

# Import the PGD attacker from the robustness package
from robustness import attacker

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def setup_attack_kwargs(config):
    """
    Set up PGD attack parameters based on the configuration.
    """
    attack_kwargs = {
        'constraint': config['test']['attacker']['constraint'],  # e.g., 'linf'
        'eps': config['test']['attacker']['epsilon'],
        'step_size': config['test']['attacker']['step_size'],
        'iterations': config['test']['attacker']['iterations'],
        'targeted': config['test']['attacker']['targeted'],
        'custom_loss': None
    }
    return attack_kwargs

def get_target(labels):
    """
    Choose a target class that is different from the true label.
    """
    a = random.randint(0, 9)
    while a == labels[0]:
        a = random.randint(0, 9)
    return torch.tensor([a])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ACGAN with PGD adversarial attack")
    parser.add_argument('--config', type=str, default='cfg/cifar_config_acgan.json',
                        help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = config['dataset']
    experiment_name = config['experiment_name']

    # Directory where the trained models are saved.
    model_dir = os.path.join("models", experiment_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load your dataset; use a batch size of 100 for splitting, then create a per-sample loader.
    dataset = get_dataset(dataset_name)
    _, test_dataset = dataset.make_loaders(workers=4, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_dataset.dataset, batch_size=1, shuffle=True)

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

    # Load the previously saved generator and discriminator.
    gan.load_model(model_dir)
    gan.generator.eval()
    gan.discriminator.eval()

    # Set up PGD attack parameters.
    attack_kwargs = setup_attack_kwargs(config)
    pgd_attacker = attacker.Attacker(model, dataset).to(device)

    clean_losses = []
    adv_losses = []
    total = 0
    correct = 0

    print("Starting ACGAN adversarial evaluation using PGD ...")
    for i, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):

        # Only process images that the model correctly classifies.
        outputs = model(images.to(device))
        _, pred = outputs.max(1)
        if pred.item() != labels.item():
            continue
        total += 1

        # Compute reconstruction loss on the clean image.
        _, loss_info = gan.reconstruct_loss(images, labels, lr=0.01, steps=100)
        clean_losses.append(loss_info[0])

        # Generate a target label (different from the true label) and create an adversarial example.
        target_label = get_target(labels)
        adv_images = pgd_attacker(images.to(device), target_label.to(device), True, **attack_kwargs)
        # Optionally normalize the adversarial images using your dataset's normalization.
        adv_images = dataset.normalize(adv_images)

        # Pass the adversarial image through the discriminator and check its auxiliary prediction.
        aux_prob, aux_out = gan.discriminator(adv_images)
        _, aux_pred = aux_out.max(1)
        # Calculate s_d as log(aux_prob) + log(aux_out for the target label)
        anomaly_score = torch.log(aux_prob) + torch.log(aux_out[:, target_label.item()])
        if aux_pred.item() == labels.item():
            correct += 1



        # Compute the reconstruction loss on the adversarial image.
        _, loss_info_adv = gan.reconstruct_loss(adv_images, target_label, lr=0.01, steps=100)
        adv_losses.append(loss_info_adv[0])

        if i >= config.get("num_test_samples", 400):
            break

    adv_det_acc = (correct / total * 100) if total > 0 else 0
    print(f"Processed {total} samples.")
    print(f"Adversarial detection (discriminator auxiliary) accuracy: {adv_det_acc:.2f}%")

    # Save the clean and adversarial loss values for further analysis.
    results_dir = os.path.join("results", experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "clean_loss.npy"), np.array(clean_losses))
    np.save(os.path.join(results_dir, "adv_loss.npy"), np.array(adv_losses))
    print(f"Results saved in {results_dir}")
