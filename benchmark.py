#!/usr/bin/env python3
import os
import torch
import argparse
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

# Import dataset and evaluation utilities.
from src.dataset_utils import get_dataset
from src.detector import UninformedStudents
from src.model_utils import resnet18_classifier
from src.eval_utils import *
from src.misc_utils import load_config, setup_attack_kwargs, get_targeted
from src.attacks import PGD

# Import ACGAN modules.
from ACGAN.GAN.acgan_1 import ACGAN
from ACGAN.GAN.acgan_res import ACGAN_Res
from ACGAN.attacks.cw import CW
from ACGAN.attacks.FGSM import FGSM

from mahalanobis.detector import MahalanobisDetector, LIDDetector

# Import PGD attacker.
from robustness import attacker

def init_detector(config, device, target_model, dataset):

    num_classes = 1000 if config['dataset'] == 'imagenet' else 10
    if config['type'] == 'uninformed_students':
        detector = UninformedStudents(config, device=device)
        detector.load(config['path'])
        detector.eval()
        detector.to(device)

    elif config['type'] == 'acgan':
        im_channel = 3  # resnet18 expects 3-channel images
        in_dim = 500

        if (config['dataset'] == 'mnist'):
            im_channel = 1
            detector = ACGAN(
                in_dim=50,
                class_dim=10,
                g_filters=[384, 192, 96, 48, im_channel],
                g_strides=[1, 2, 2, 2],
                d_filters=[16, 32, 64, 128, 256, 512],
                d_strides=[2, 1, 2, 1, 2, 1],
                CNN=target_model
            )
        else:
            detector = ACGAN_Res(in_dim=in_dim, class_dim=num_classes, CNN=target_model)
        detector.load_model(config['path'])
        detector.generator.eval()
        detector.discriminator.eval()
    
    elif config['type'] == 'mahalanobis':
        detector = MahalanobisDetector(target_model, device='cuda', net_type='resnet', dataset=dataset)
        detector.load(config['path'])
        detector.model.eval()

    elif config['type'] == 'lid':
        detector = LIDDetector(target_model, device='cuda', dataset=dataset)
        detector.load(config['path'])
        detector.model.eval()

    else:
        raise ValueError(f"Unknown detector type: {config['type']}")
    
    return detector

def main(config):
    # Reproducibility: seed everything if `config['seed']` is set ──
    seed = config.get('seed', None)
    if seed is not None:
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Prepare to collect pAUC results per attack ──
    attack_names = []
    targeted_list = []
    epsilon_list = []
    constraint_list = []
    acc_list = []
    results_dir = os.path.join("results", config['experiment_name'])
    n_samples = config['test']['samples']
    ensure_succesful_attack = config['test']['ensure_succesful_attack']

    # Create directories if they do not exist
    os.makedirs(results_dir, exist_ok=True)

    # Use GPU for the classification network and PGD attacker.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset and create a per-sample loader.
    dataset = get_dataset(config['dataset'])
    test_loader = dataset.make_loaders(workers=0, batch_size=1, only_test=True)
    norm = dataset.normalize

    # Load the pretrained ResNet18 classifier.
    target_model = resnet18_classifier(device, config['dataset'], config['target_model_path'])

    # Initialize pgd attacker.
    pgd_attacker = attacker.Attacker(target_model, dataset).to(device)

    # Initialize the detector list
    detectors = {}
    for detector_config in config['detectors']:
        aux_config = config.copy()
        aux_config.pop('detectors', None)
        aux_config = {**detector_config, **aux_config}
        detectors[detector_config['type']] = init_detector(aux_config, device, target_model, dataset)

    # -------------------------------
    # 1. Natural (clean) evaluation loop.
    # -------------------------------
    nat_as = {detector: [] for detector in detectors.keys()}
    nat_accuracy = 0

    print("Starting evaluation on natural (clean) images ...")

    i = 0
    pbar = tqdm(total=n_samples, desc="Evalutaing on Natural Images")
    for images, labels in test_loader:
        y = target_model(norm(images.to(device))).detach().cpu()

        with torch.no_grad():
            # if incorrect prediction, skip the sample
            if ensure_succesful_attack and (y.argmax(1) != labels).sum().item() > 0:
                    continue
            
            # Calculate the accuracy on natural images.
            nat_accuracy += (y.argmax(1) == labels).sum().item()/n_samples
            
            for type, detector in detectors.items():
                if type == 'acgan':
                    # Calculate AS for ACGAN
                    as_ACGAN = sd_statistic(detector.discriminator(images.to(device)), labels)
                    nat_as[type].append(as_ACGAN.detach().cpu().item())
                else:
                    with torch.enable_grad(): # MahalanobisDetector requires gradients
                        nat_as[type].append(detector(norm(images.to(device)), labels).detach().cpu().item())

        pbar.update(1)
        i += 1
        if i >= n_samples:
            break

    for metric in nat_as:
        nat_as[metric] = np.array(nat_as[metric])

    # -------------------------------s
    # 2. Adversarial evaluation loop.
    # -------------------------------

    det_acc_list = {detector: [] for detector in detectors.keys()}
    pAUC_list = {detector: [] for detector in detectors.keys()}
    # Iterate over the list of attack configurations.
    for attack_config in config['test']['attack_list']:

        # Initialize lists to store the anomaly scores for adversarial images.
        adv_as = {detector: [] for detector in detectors.keys()}
        adv_accuracy = 0

        # See if attack is targeted
        targeted, str_targeted = get_targeted(attack_config)

        # Create individual directories for each attack configuration.
        results_dir_attack = results_dir + "/" + attack_config['type'] + "_" + str_targeted + "_" + attack_config['constraint'] + "_" + str(attack_config['epsilon']) 
        os.makedirs(results_dir_attack, exist_ok=True)
        os.makedirs(results_dir_attack + "/acgan", exist_ok=True)
        os.makedirs(results_dir_attack + "/uninformed_students", exist_ok=True)

        # Initialize the attacker (or attack args for PGD).
        if (attack_config['type'] == 'pgd'):
             attack_kwargs = setup_attack_kwargs(attack_config)    
        elif (attack_config['type'] == 'cw'):
            cw = CW(model=target_model, norm=norm, kappa=attack_config['kappa'], steps=attack_config['iterations'], lr=attack_config['step_size'], targeted=attack_config['targeted'])
        elif (attack_config['type'] == 'fgsm'):
            fgsm = FGSM(model=target_model, norm=norm, epsilon=attack_config['epsilon'], targeted=attack_config['targeted'])

        i = 0
        pbar = tqdm(total=n_samples, desc="Adversarial Evaluation with " + attack_config['type'] + "_" + attack_config['constraint'] + "_" + str(attack_config['epsilon']))
        for images, labels in test_loader:
            
            # if incorrect prediction, skip the sample
            if ensure_succesful_attack and (target_model(norm(images.to(device))).argmax(1) != labels.to(device)).sum().item() > 0:
                continue

            # Choose target labels depending on the attack type.
            target_labels = get_target(labels) if targeted else labels

            if (attack_config['type'] == 'pgd'):
                adv_images = pgd_attacker(images.to(device), target_labels.to(device), attack_config["targeted"], **attack_kwargs)
            elif (attack_config['type'] == 'cw'):
                adv_images = cw.attack(images.to(device), target_labels)
            elif (attack_config['type'] == 'fgsm'):
                adv_images = fgsm.attack(images.to(device), target_labels)
            else:
                raise ValueError("Invalid attack type: " + attack_config['type'])

            with torch.no_grad():
                # Compute classification accuracy on adversarial images (using the classifier on GPU).
                y = target_model(norm(adv_images).to(device)).detach().cpu()

                if ensure_succesful_attack and (y.argmax(1) != labels).sum().item() < len(labels):
                    continue
                        
                # Calculate the accuracy on adversarial images.
                adv_accuracy += (y.argmax(1) == labels).sum().item()/n_samples
            
                for type, detector in detectors.items():
                    if type == 'acgan':
                        # Calculate AS for ACGAN
                        as_ACGAN = sd_statistic(detector.discriminator(adv_images), y.argmax(1))
                        adv_as[type].append(as_ACGAN.detach().cpu().item())
                    else:
                        with torch.enable_grad(): # MahalanobisDetector requires gradients
                            adv_as[type].append(detector(norm(adv_images).to(device), labels).detach().cpu().item())

            pbar.update(1)
            i += 1
            if i >= n_samples:
                break

        # Convert lists to numpy arrays for easier processing.
        for metric in adv_as:
            adv_as[metric] = np.array(adv_as[metric])

        # -------------------------------
        # 3. Compute detection metrics.
        # -------------------------------
        # Use the 90th percentile of normalized natural anomaly scores as the threshold.
        for detector in detectors:
            threshold = torch.quantile(torch.tensor(nat_as[detector]), 0.90).item()
            detected = np.sum(adv_as[detector] > threshold)
            det_acc_list[detector].append((detected / len(adv_as[detector]) * 100) if len(adv_as[detector]) > 0 else 0)
            pAUC_list[detector].append(partial_auc(nat_as[detector].tolist(), adv_as[detector].tolist()))

        targeted, str_targeted = get_targeted(attack_config)
        attack_names.append(attack_config['type'])
        targeted_list.append(str_targeted)
        constraint_list.append(str(attack_config['constraint']))
        epsilon_list.append(str(attack_config['epsilon']))
        acc_list.append(adv_accuracy * 100)

    # -------------------------------s
    # 4. Save and display results.
    # -------------------------------
    headers = ["Detector"] + attack_names
    rows = [
        ["Targeted"] + targeted_list,
        ["Constraint"] + constraint_list,
        ["Epsilon"] + epsilon_list
        ]
    for detector in detectors.keys():
        rows.append([detector] + pAUC_list[detector])
    rows.append([nat_accuracy*100] + acc_list)
    
    return headers, rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ACGAN with PGD adversarial attack and compute anomaly score metrics")
    parser.add_argument('--config', type=str, default='cfg/mnist_benchmark.json', help='Path to the configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)

    headers, rows = main(config)

    data = [headers] + rows
    transposed = list(map(list, zip(*data)))
    new_headers, *new_rows = transposed

    print("\n\n--- Full Results ---\n")
    table_str = tabulate(
        new_rows,
        headers=new_headers,
        tablefmt='latex_raw',
        stralign='center'
    )
    print(table_str)

    base = os.path.splitext(args.config)[0]
    out_fname = f"results{base[3:]}.txt"
    with open(out_fname, 'w') as f:
        f.write(table_str + "\n")
    print(f"\nSaved aggregated results to {out_fname}")