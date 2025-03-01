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
from src.model_utils import resnet18_classifier, model_paths
import time

def initialize_detector(config, dataset, device):
    """
    Initialize the appropriate detector model based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        dataset: Dataset object
        device: PyTorch device
    
    Returns:
        detector: Initialized detector model
    """
    method = config['method']
    
    if method == 'STFPM':
        detector = STFPM(dataset, device, config['train']['learning_rate'])
    elif method == 'ClassConditionalUninformedStudents':
        detector = ClassConditionalUninformedStudents(
            config['num_students'], 
            dataset, 
            patch_size=config['patch_size'], 
            device=device
        )
    elif method == 'UninformedStudents':
        detector = UninformedStudents(
            config['num_students'], 
            dataset, 
            patch_size=config['patch_size'], 
            device=device
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return detector

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def train(steps, device, norm, writer, train_loader, detector, save_path, test_loader=None, val_interval=None):
    i = 0
    epoch = 1
    best_pAUC = 0
    # Train the model for the specified number of epochs
    start_time = time.time()
    i = 0
    while i < steps:
        epoch_start = time.time()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = norm(images).to(device)
            loss = detector.train_step(images)
            writer.add_scalar('Loss/train', loss.item(), i)
            i += 1

            if val_interval is not None and i % val_interval == 0:
                if test_loader is None:
                    print(f"Epoch {epoch} time: {epoch_time:.1f}s, Est. time left: {estimated_time_left/60:.1f}min")
                else:
                    results = test(detector, test_loader, device, norm)
                    if results['pAUC'] > best_pAUC:
                        best_pAUC = results['pAUC']
                        detector.save(save_path+'_best')
                        print("Model saved at", save_path+'_best')

                    writer.add_scalar('Metrics/pAUC', results['pAUC'], i)
                    
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
            results = test(detector, test_loader, device, norm)
            if results['pAUC'] > best_pAUC:
                best_pAUC = results['pAUC']
                detector.save(save_path+'_best')
                print("Model saved at", save_path+'_best')

            writer.add_scalar('Metrics/pAUC', results['pAUC'], i)
            print(f"Epoch: {epoch}, pAUC: {results['pAUC']}, time: {epoch_time:.1f}s, Est. time left: {estimated_time_left/60:.1f}min")
            epoch += 1

def test(detector, test_loader, device, norm, n_samples=500, epsilon=0.05):
    dataset = test_loader.dataset
    target_model = resnet18_classifier(device, dataset.ds_name, path=model_paths[dataset.ds_name])

    e_list = []
    u_list = []
    nat_as = []  # Will store anomaly scores for clean images.
    nat_accuracy = 0

    i = 0
    for images, labels in test_loader:
        # For classification, move inputs to GPU.
        y = target_model(norm(images.to(device))).detach().cpu()
        nat_accuracy += (y.argmax(1) == labels).sum().item()/n_samples

        if isinstance(detector, UninformedStudents):
            regression_error, predictive_uncertainty = detector.forward(norm(images.to(device)), labels)
            # Append the values to the anomaly score list
            e_list.append(regression_error)
            u_list.append(predictive_uncertainty)
        else:
            anomaly_score = detector.forward(norm(images.to(device)))
            nat_as.append(anomaly_score.item())

        i += 1
        if i >= n_samples:
            break

    if isinstance(detector, UninformedStudents):
        # Calculate the mean and standard deviation of e_list and u_list
        detector.e_mean = torch.tensor(e_list).mean().item()
        detector.e_std = torch.tensor(e_list).std().item()
        detector.v_mean = torch.tensor(u_list).mean().item()
        detector.v_std = torch.tensor(u_list).std().item()

        # Normalize e_list and u_list
        e_list = [(e - detector.e_mean) / detector.e_std for e in e_list]
        u_list = [(u - detector.v_mean) / detector.v_std for u in u_list]

        # Compute the anomaly score list
        nat_as = [e + u for e, u in zip(e_list, u_list)]
        
    nat_as = np.array(nat_as)

    # Calculate the top 1% quantile 
    threshold = torch.quantile(torch.tensor(nat_as), 0.90).item()

    # Initialize lists to store the anomaly scores for adversarial images.
    adv_as = []
    adv_accuracy = 0

    # Initialize the attacker
    fgsm = FGSM(model=target_model, norm=norm, epsilon=epsilon, targeted=-1)

    processed = 0
    for images, labels in test_loader:
        # if incorrect prediction, skip the sample
        if (target_model(norm(images.to(device))).argmax(1) != labels.to(device)).sum().item() > 0:
            continue

        # Choose target labels depending on the attack type.
        adv_images = fgsm.attack(images.to(device), labels)

        # Compute classification accuracy on adversarial images (using the classifier on GPU).
        y = target_model(norm(adv_images).to(device)).detach().cpu()

        if (y.argmax(1) != labels).sum().item() < len(labels):
            continue
                
        # Calculate the accuracy on adversarial images.
        adv_accuracy += (y.argmax(1) == labels).sum().item()/n_samples

        if isinstance(detector, UninformedStudents):
            regression_error, predictive_uncertainty = detector.forward(norm(adv_images).to(device), y.argmax(1))
            anomaly_score = (regression_error - detector.e_mean) / detector.e_std + (predictive_uncertainty - detector.v_mean) / detector.v_std
            adv_as.append(anomaly_score)
        else:
            anomaly_score = detector.forward(norm(adv_images).to(device))
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

def main(args):

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
    train_loader, test_loader = dataset.make_loaders(batch_size=batch_size, workers=4, only_train=True)
    test_loader.dataset.ds_name = dataset_name

    # Then call the function to initialize the detector
    detector = initialize_detector(config, dataset, device)

    train(steps, device, dataset.normalize, writer, train_loader, detector, save_path, test_loader, val_interval=1000)
    detector.save(save_path)
    print("Model saved at", save_path)

    #detector.load(save_path)

    results = test(detector, dataset, test_loader, device, 1000)

    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the detector model.")
    parser.add_argument('--config', type=str, default='cfg/cifar_train_us.json', help='Path to the configuration file.')
    args = parser.parse_args()

    main(args)