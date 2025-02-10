#!/usr/bin/env python3
import os
import json
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
from torch import nn
import torchvision.models as models

# Import dataset and evaluation utilities.
from src.dataset_utils import get_dataset
from src.eval_utils import partial_auc, save_results

# Import ACGAN modules.
from ACGAN.GAN.acgan_1 import ACGAN

# Import PGD attacker.
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
        'constraint': config['test']['attacker']['constraint'],
        'eps': config['test']['attacker']['epsilon'],
        'step_size': config['test']['attacker']['step_size'],
        'iterations': config['test']['attacker']['iterations'],
        'targeted': config['test']['attacker']['targeted'],
        'custom_loss': None
    }
    return attack_kwargs

def get_target(labels):
    """
    Choose a target class different from the true label.
    """
    a = random.randint(0, 9)
    while a == labels[0]:
        a = random.randint(0, 9)
    return torch.tensor([a])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test ACGAN with PGD adversarial attack and compute anomaly score metrics"
    )
    parser.add_argument('--config', type=str, default='cfg/cifar_config_acgan.json',
                        help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = config['dataset']
    experiment_name = config['experiment_name']

    # Directories for models and results.
    model_dir = os.path.join("models", experiment_name)
    results_dir = os.path.join("results", experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    # Use GPU for the classification network and PGD attacker.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset and create a per-sample loader.
    dataset = get_dataset(dataset_name)
    _, test_dataset = dataset.make_loaders(workers=4, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_dataset.dataset, batch_size=1, shuffle=True)

    # Load the pretrained ResNet18 classifier.
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # For CIFAR-10.
    model_weights_path = os.path.join('models', 'cifar_epsilon_0.1/teacher.pth')#'resnet18_cifar.pth')
    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    model.eval()

    # Initialize ACGAN.
    im_channel = 3
    gan = ACGAN(
        in_dim=50,
        class_dim=10,
        g_filters=[384, 192, 96, 48, im_channel],
        g_strides=[1, 2, 2, 2],
        d_filters=[16, 32, 64, 128, 256, 512],
        d_strides=[2, 1, 2, 1, 2, 1],
        CNN=model
    )
    gan.load_model(model_dir)
    gan.generator.eval()
    gan.discriminator.eval()

    # Set up PGD attacker.
    attack_kwargs = setup_attack_kwargs(config)
    pgd_attacker = attacker.Attacker(model, dataset).to(device)

    n_samples = config['test']['samples']

    # -------------------------------
    # 1. Natural (clean) evaluation loop.
    # -------------------------------
    nat_scores = []  # Will store anomaly scores for clean images.
    nat_accuracy = 0

    print("Starting evaluation on natural (clean) images ...")
    for i, (images, labels) in enumerate(tqdm(test_loader, desc="Natural Evaluation", total=n_samples)):
        #images = dataset.normalize(images.to(device))
        # For classification, move inputs to GPU.
        y = model(dataset.normalize(images.to(device))).detach().cpu()
        nat_accuracy += (y.argmax(1) == labels).sum().item()/n_samples

        # For anomaly scoring, use GAN's discriminator.
        # Ensure the inputs are on CPU since ACGAN methods require CPU.
        aux_prob, aux_out = gan.discriminator(images.to(device))

        # Compute the anomaly score using the true label.
        true_label = labels.item()
        anomaly_score = torch.log(aux_prob) + torch.log(aux_out[:, true_label])
        nat_scores.append(anomaly_score.item())

        if i >= n_samples:
            break

    nat_scores = np.array(nat_scores)
    nat_scores[np.isneginf(nat_scores)] = -25
    nat_scores = -nat_scores  # Invert the scores for adversarial images.

    # Calculate the top 1% quantile 
    threshold = torch.quantile(torch.tensor(nat_scores), 0.90).item()

    # -------------------------------
    # 2. Adversarial evaluation loop.
    # -------------------------------
    adv_scores = []  # Will store anomaly scores for adversarial images.
    adv_accuracy = 0

    print("Starting evaluation on adversarial images ...")
    for i, (images, labels) in enumerate(tqdm(test_loader, desc="Adversarial Evaluation", total=n_samples)):
        #outputs = model(dataset.normalize(images.to(device)))
        #_, pred = outputs.max(1)
        # Only consider images that are correctly classified.
        #if pred.item() != labels.to(device).item():
        #    continue

        # Generate a target label (different from the true label).
        target_label = get_target(labels)

        # Generate adversarial examples using the PGD attacker.
        adv_images = pgd_attacker(images.to(device), target_label.to(device), True, **attack_kwargs)
        # Optionally normalize the adversarial images.

        # Compute classification accuracy on adversarial images (using the classifier on GPU).
        y = model(dataset.normalize(adv_images).to(device)).detach().cpu()
        adv_accuracy += (y.argmax(1) == labels).sum().item()/n_samples

        aux_prob, aux_out = gan.discriminator(adv_images)
        anomaly_score_adv = torch.log(aux_prob) + torch.log(aux_out[:, target_label.item()])
        adv_scores.append(anomaly_score_adv.item())

        if i >= n_samples:
            break

    # Normalize adversarial anomaly scores using natural score statistics.
    adv_scores = np.array(adv_scores)
    adv_scores[np.isneginf(adv_scores)] = -25
    adv_scores = -adv_scores  # Invert the scores for adversarial images.

    # -------------------------------
    # 3. Compute detection metrics.
    # -------------------------------
    # Use the 90th percentile of normalized natural anomaly scores as the threshold.
    detected_adv = np.sum(adv_scores > threshold)
    adv_detection_accuracy = (detected_adv / len(adv_scores) * 100) if len(adv_scores) > 0 else 0

    # Compute partial AUC (pAUC) using the provided utility.
    pAUC = partial_auc( nat_scores.tolist(), adv_scores.tolist())

    # -------------------------------
    # 4. Save and display results.
    # -------------------------------
    results = {
        "nat_list": nat_scores.tolist(),
        "adv_list": adv_scores.tolist(),
        "threshold": threshold,
        "natural_accuracy": nat_accuracy,
        "adversarial_accuracy": adv_accuracy,
        "adversarial_detection_accuracy": adv_detection_accuracy,
        "pAUC": pAUC
    }
    save_results(results_dir, results, range=(0, 25))

    print(f"Results saved in {results_dir}")
    print("Detailed results:")
    print(f"  Natural Accuracy: {nat_accuracy:.2f}%")
    print(f"  Adversarial Accuracy: {adv_accuracy:.2f}%")
    print(f"  Adversarial Detection Accuracy: {adv_detection_accuracy:.2f}%")
    print(f"  Threshold (90th percentile): {threshold:.4f}")
    print(f"  Partial AUC: {pAUC:.4f}")
