#!/usr/bin/env python3
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

# Import dataset and evaluation utilities.
from src.dataset_utils import get_dataset
from src.detector import UninformedStudents
from src.model_utils import resnet18_classifier
from src.eval_utils import *
from src.misc_utils import load_config


from src.attacks import *


def main(config):
    # Directories for models and results.
    results_dir = os.path.join("results", config['experiment_name'])
    n_samples = 500
    epsilon = 0.1
    num_steps = 100
    lr = 2.5*epsilon/num_steps # Step size for PGD attack, as defined in the paper.
    k= 1.0

    # Use GPU for the classification network and PGD attacker.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset and create a per-sample loader.
    dataset = get_dataset(config['dataset'])
    test_loader = dataset.make_loaders(workers=0, batch_size=1, only_test=True)
    norm = dataset.normalize

    # Load the pretrained ResNet18 classifier.
    target_model = resnet18_classifier(device, config['dataset'], config['target_model_path'])

    # Initialize Uninformed Students
    detector = UninformedStudents(config, device=device)
    detector.load(config['uninformed_students_path'])
    detector.eval()
    detector.to(device)
    

    # Set up attackers.
    pgd = PGD(model=target_model, norm=norm, device=device, epsilon=epsilon, alpha=lr, steps=num_steps)
    pgd_wb = PGDWhiteBox(model=target_model, norm=norm, device=device, detector=detector, k=k, epsilon=epsilon, alpha=lr, steps=num_steps)

    nat_as = []  # Will store anomaly scores for clean images.
    adv_as = []  # Will store anomaly scores for adversarial images.
    adv_wb_as = []  # Will store anomaly scores for adversarial images (white-box).
    nat_accuracy = []
    adv_accuracy = []
    adv_wb_accuracy = []

    print("Starting evaluation ...")

    i = 0
    pbar = tqdm(total=n_samples, desc="Evaluating")
    for images, labels in test_loader:
        target_labels = get_target(labels)
        
        # Natural images evaluation
        y = target_model(norm(images.to(device))).detach().cpu()
        nat_accuracy.append((y.argmax(1) == labels).sum().item())

        # Calculate the anomaly score for natural images using Uninformed Students.
        re, pu = detector.forward(norm(images.to(device)))
        AS = (re - detector.e_mean) / detector.e_std + (pu - detector.v_mean) / detector.v_std
        nat_as.append(AS)
        
        # Adversarial images evaluation
        adv_images = pgd.attack(images.to(device), target_labels.to(device))
        y = target_model(norm(adv_images).to(device)).detach().cpu()
        adv_accuracy.append((y.argmax(1) == labels).sum().item())

        # Calculate the anomaly score for adversarial images using Uninformed Students.
        re, pu = detector.forward(norm(adv_images.to(device)))
        AS = (re - detector.e_mean) / detector.e_std + (pu - detector.v_mean) / detector.v_std
        adv_as.append(AS)

        # Adversarial images evaluation (white-box)
        adv_images_wb = pgd_wb.attack(images.to(device), target_labels.to(device))
        y = target_model(norm(adv_images_wb).to(device)).detach().cpu()
        adv_wb_accuracy.append((y.argmax(1) == labels).sum().item())

        # Calculate the anomaly score for adversarial images using Uninformed Students (white-box).
        re, pu = detector.forward(norm(adv_images_wb.to(device)))
        AS = (re - detector.e_mean) / detector.e_std + (pu - detector.v_mean) / detector.v_std
        adv_wb_as.append(AS)


        pbar.update(1)
        i += 1
        if i >= n_samples:
            break


    nat_as = np.array(nat_as)
    adv_as = np.array(adv_as)
    adv_wb_as = np.array(adv_wb_as)
    nat_accuracy = np.array(nat_accuracy)
    adv_accuracy = np.array(adv_accuracy)
    adv_wb_accuracy = np.array(adv_wb_accuracy)

    results = {
        'nat_as': nat_as,
        'adv_as': adv_as,
        'adv_wb_as': adv_wb_as,
        'nat_accuracy': nat_accuracy,
        'adv_accuracy': adv_accuracy,
        'adv_wb_accuracy': adv_wb_accuracy,
        'classifier_nat_acc': (nat_accuracy.sum() / len(nat_accuracy) * 100) if len(nat_accuracy) > 0 else 0
    }

    # Calculate Acc at 10% false positive rate (FPR) for Uninformed Students
    threshold = torch.quantile(torch.tensor(nat_as), 0.90).item()
    detected = nat_as > threshold
    accepted = np.logical_not(detected)
    results['classifier_nat_acc_filtered_10fpr'] = (sum(accepted*nat_accuracy) / sum(accepted) * 100) if sum(accepted) > 0 else 0

    detected = adv_as > threshold
    accepted = np.logical_not(detected)
    results['detector_adv_acc_10fpr'] = (detected.sum() / len(detected) * 100) if len(detected) > 0 else 0
    results['classifier_adv_acc'] = (adv_accuracy.sum()  / len(adv_accuracy) * 100) if len(adv_accuracy) > 0 else 0
    results['classifier_adv_acc_filtered_10fpr'] = (sum(accepted*adv_accuracy) / sum(accepted) * 100) if sum(accepted) > 0 else 0

    detected = adv_wb_as > threshold
    accepted = np.logical_not(detected)
    results['detector_adv_wb_acc_10fpr'] = (detected.sum()  / len(detected) * 100) if len(detected) > 0 else 0
    results['classifier_adv_wb_acc'] = (adv_wb_accuracy.sum()  / len(adv_wb_accuracy) * 100) if len(adv_wb_accuracy) > 0 else 0
    results['classifier_adv_wb_acc_filtered_10fpr'] = (sum(accepted*adv_wb_accuracy) / sum(accepted) * 100) if sum(accepted) > 0 else 0

    # Calculate Acc at 1% false positive rate (FPR) for Uninformed Students
    threshold = torch.quantile(torch.tensor(nat_as), 0.99).item()
    detected = nat_as > threshold
    accepted = np.logical_not(detected)
    results['classifier_nat_acc_filtered_1fpr'] = (sum(accepted*nat_accuracy) / sum(accepted) * 100) if sum(accepted) > 0 else 0

    detected = adv_as > threshold
    accepted = np.logical_not(detected)
    results['detector_adv_acc_1fpr'] = (detected.sum()  / len(detected) * 100) if len(detected) > 0 else 0
    results['classifier_adv_acc_filtered_1fpr'] = (sum(accepted*adv_accuracy) / sum(accepted) * 100) if sum(accepted) > 0 else 0

    detected = adv_wb_as > threshold
    accepted = np.logical_not(detected)
    results['detector_adv_wb_acc_1fpr'] = (detected.sum()  / len(detected) * 100) if len(detected) > 0 else 0
    results['classifier_adv_wb_acc_filtered_1fpr'] = (sum(accepted*adv_wb_accuracy) / sum(accepted) * 100) if sum(accepted) > 0 else 0

    # print("Results:")
    # for key, value in results.items():
    #     print(f"{key}: {value}")

        # Calculate the results table
    nat_as_t = torch.tensor(nat_as)
    th10 = torch.quantile(nat_as_t, 0.90).item()
    th1  = torch.quantile(nat_as_t, 0.99).item()
    det_nat_acc_10 = ((nat_as_t <= th10).sum().item() / len(nat_as) * 100)
    det_nat_acc_1  = ((nat_as_t <= th1 ).sum().item() / len(nat_as) * 100)

    # Build the rows for the table
    rows = [
        ["Nat", "Classifier",
         results['classifier_nat_acc'],
         det_nat_acc_10,
         results['classifier_nat_acc_filtered_10fpr'],
         det_nat_acc_1,
         results['classifier_nat_acc_filtered_1fpr'],
        ],
        ["PGD", "Classifier",
         results['classifier_adv_acc'],
         results['detector_adv_acc_10fpr'],
         results['classifier_adv_acc_filtered_10fpr'],
         results['detector_adv_acc_1fpr'],
         results['classifier_adv_acc_filtered_1fpr'],
        ],
        ["PGD", "Classifier+Detector",
         results['classifier_adv_wb_acc'],
         results['detector_adv_wb_acc_10fpr'],
         results['classifier_adv_wb_acc_filtered_10fpr'],
         results['detector_adv_wb_acc_1fpr'],
         results['classifier_adv_wb_acc_filtered_1fpr'],
        ],
    ]

    headers = [
        "Attack",
        "Attack Target",
        "Classifier Acc",
        "Detector Acc (10% FPR)",
        "Classifier Acc Without detected attacks",
        "Detector Acc (1% FPR)",
        "Classifier Acc Without detected attacks",
    ]

    # Print the results table
    print("\n\n\n")
    print(tabulate(rows, headers=headers, floatfmt=".1f"))
    print("\n\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ACGAN with PGD adversarial attack and compute anomaly score metrics")
    parser.add_argument('--config', type=str, default='cfg/cifar_benchmark.json', help='Path to the configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)

    main(config)