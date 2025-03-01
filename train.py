import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

from src.detector import STFPM, ClassConditionalUninformedStudents, UninformedStudents
from src.dataset_utils import get_dataset
from ACGAN.attacks.FGSM import FGSM
from src.eval_utils import *

import numpy as np
from torch import nn
import torchvision.models as models
from src.model_utils import resnet18_classifier
import time

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def train(steps, device, writer, train_loader, detector, test_loader=None):
    i = 0
    epoch = 1
    best_pAUC = 0
    # Train the model for the specified number of epochs
    start_time = time.time()
    i = 0
    while i < steps:
        epoch_start = time.time()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            loss = detector.train_step(images)
            writer.add_scalar('Loss/train', loss.item(), i)
            i += 1
            
            if i >= steps:
                break

        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        steps_left = steps - i
        estimated_time_left = (steps_left * elapsed_time) / i if i > 0 else 0
        
        print(f"Progress: {i}/{steps} steps ({(i/steps)*100:.1f}%)")
        if test_loader is None:
            print(f"Epoch {epoch} time: {epoch_time:.1f}s, Est. time left: {estimated_time_left/60:.1f}min")
        else:
            results = test(detector, test_loader, device)
            if results['pAUC'] > best_pAUC:
                best_pAUC = results['pAUC']
                detector.save(save_path+'_best')
                print("Model saved at", save_path+'_best')

            writer.add_scalar('Metrics/pAUC', results['pAUC'], i)
            print(f"Epoch: {epoch}, pAUC: {results['pAUC']}, time: {epoch_time:.1f}s, Est. time left: {estimated_time_left/60:.1f}min")
            epoch += 1

def test(detector, test_loader, device, n_samples=500, epsilon=0.1):
    target_model = resnet18_classifier(device, test_loader.dataset.ds_name, 'models/ckpt.pth')

    nat_as = []  # Will store anomaly scores for clean images.
    nat_accuracy = 0

    i = 0
    for images, labels in test_loader:
        # For classification, move inputs to GPU.
        y = target_model(dataset.normalize(images.to(device))).detach().cpu()
        nat_accuracy += (y.argmax(1) == labels).sum().item()/n_samples

        # Calculate AS for Uninformed Students
        anomaly_score = detector.forward(dataset.normalize(images.to(device)))

        # Append the values to the anomaly score list.
        nat_as.append(anomaly_score.item())

        i += 1
        if i >= n_samples:
            break

    nat_as = np.array(nat_as)

    # Calculate the top 1% quantile 
    threshold = torch.quantile(torch.tensor(nat_as), 0.90).item()

    # Initialize lists to store the anomaly scores for adversarial images.
    adv_as = []
    adv_accuracy = 0

    # Initialize the attacker
    fgsm = FGSM(model=target_model, norm=dataset.normalize, epsilon=epsilon, targeted=-1)

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
    parser.add_argument('--config', type=str, default='cfg/cifar_train_stfpm.json', help='Path to the configuration file.')
    args = parser.parse_args()

    # Load configuration from JSON
    config = load_config(args.config)

    # Parameters from JSON
    dataset_name = config['dataset']
    save_path = config['model_path']
    steps = config['train']['steps']
    batch_size = config['train']['batch_size']

    # Initialize the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the Tensorboard writer
    writer = SummaryWriter(comment=f"_{dataset_name}")

    # Get the dataset and create data loaders
    dataset = get_dataset(dataset_name)
    train_loader, _ = dataset.make_loaders(workers=4, batch_size=batch_size)
    _, test_loader = dataset.make_loaders(workers=4, batch_size=1)
    test_loader.dataset.ds_name = dataset_name

    # Initialize the detector model
    if config['method'] == 'STFPM':
        detector = STFPM(dataset, device, config['train']['learning_rate'])
    elif config['method'] == 'ClassConditionalUninformedStudents':
        detector = ClassConditionalUninformedStudents(config['num_students'], dataset, patch_size=config['patch_size'], device=device)
    elif config['method'] == 'UninformedStudents':
        detector = UninformedStudents(config['num_students'], dataset, patch_size=config['patch_size'], device=device)
    else:
        raise ValueError(f"Unknown method: {config['method']}")

    train(steps, device, writer, train_loader, detector, test_loader)
    detector.save(save_path)
    print("Model saved at", save_path)

    #detector.load(save_path)

    results = test(detector, dataset, test_loader, device, 1000)

    print(results)