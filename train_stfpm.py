import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

from src.detector import STFPM
from src.model_utils import extract_patches
from src.dataset_utils import get_dataset
from ACGAN.attacks.FGSM import FGSM
from src.eval_utils import *

import numpy as np
from torch import nn
import torchvision.models as models

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def train(steps, device, writer, train_loader, detector):
    i = 0
    # Train the model for the specified number of epochs
    with tqdm(total=steps, desc="Training Progress") as pbar:
        while i < steps:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)

                loss = detector.train_step(images)

                # Log the training loss
                writer.add_scalar('Loss/train', loss.item(), i)

                i += 1
                pbar.update(1)

                if i >= steps:
                    break

def test(detector, dataset, test_loader, device, n_samples=1000):
    target_model = models.resnet18(pretrained=False)
    target_model.fc = nn.Linear(target_model.fc.in_features, 10)  # For CIFAR-10.
    target_model.load_state_dict(torch.load('models/resnet18_cifar.pth'))
    target_model.to(device)
    target_model.eval()

    nat_as = []  # Will store anomaly scores for clean images.
    nat_accuracy = 0

    print("Starting evaluation on natural (clean) images ...")
    for i, (images, labels) in enumerate(tqdm(test_loader, desc="Natural Evaluation", total=n_samples)):
        # For classification, move inputs to GPU.
        y = target_model(dataset.normalize(images.to(device))).detach().cpu()
        nat_accuracy += (y.argmax(1) == labels).sum().item()/n_samples

        # Calculate AS for Uninformed Students
        anomaly_score = detector.forward(dataset.normalize(images.to(device)))

        # Append the values to the anomaly score list.
        nat_as.append(anomaly_score.item())

        if i >= n_samples:
            break

    nat_as = np.array(nat_as)

    # Calculate the top 1% quantile 
    threshold = torch.quantile(torch.tensor(nat_as), 0.90).item()

    # Initialize lists to store the anomaly scores for adversarial images.
    adv_as = []
    adv_accuracy = 0

    # Initialize the attacker
    fgsm = FGSM(model=target_model, norm=dataset.normalize, epsilon=0.05, targeted=-1)

    processed = 0
    for images, labels in test_loader:
        # if incorrect prediction, skip the sample
        if (target_model(dataset.normalize(images.to(device))).argmax(1) != labels.to(device)).sum().item() > 0:
            continue

        # Choose target labels depending on the attack type.
        adv_images = fgsm.attack(images.to(device), labels)

        # Compute classification accuracy on adversarial images (using the classifier on GPU).
        y = target_model(dataset.normalize(adv_images).to(device)).detach().cpu()

        if (y.argmax(1) != labels).sum().item() < len(labels):
            continue
                
        # Calculate the accuracy on adversarial images.
        adv_accuracy += (y.argmax(1) == labels).sum().item()/n_samples

        anomaly_score = detector.forward(dataset.normalize(adv_images).to(device))
        adv_as.append(anomaly_score.cpu().item())

        processed += 1
        if processed >= n_samples:
            break

    adv_as = np.array(adv_as)
    detected = np.sum(adv_as > threshold)
    acc = (detected / len(adv_as) * 100) if len(adv_as) > 0 else 0

    # Compute partial AUC (pAUC) using the provided utility.
    pAUC = partial_auc(nat_as.tolist(), adv_as.tolist())

    # -------------------------------s
    # 4. Save and display results.
    # -------------------------------
    results = {
        "threshold": threshold,
        "natural_accuracy": nat_accuracy,
        "adversarial_accuracy": adv_accuracy,
        "adversarial_detection_accuracy": acc,
        "pAUC": pAUC
    }
    
    return results

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate the detector model.")
    parser.add_argument('--config', type=str, default='cfg/cifar_train.json', help='Path to the configuration file.')
    args = parser.parse_args()

    # Load configuration from JSON
    config = load_config(args.config)


    # Parameters from JSON
    dataset_name = 'cifar'
    save_path = 'models/cifar_stfpm'
    steps = 100000
    lr = 0.4
    batch_size = 32

    # Initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the Tensorboard writer
    writer = SummaryWriter(comment=f"_{dataset_name}")

    # Get the dataset and create data loaders
    dataset = get_dataset(dataset_name)
    train_loader, _ = dataset.make_loaders(workers=4, batch_size=batch_size)
    _, test_loader = dataset.make_loaders(workers=4, batch_size=1)

    # Initialize the detector model
    detector = STFPM(dataset, device, lr)

    #train(steps, device, writer, train_loader, detector)

    detector.load(save_path)

    n_samples = 1000

    results = test(detector, dataset, test_loader, device, n_samples)

    print(results)