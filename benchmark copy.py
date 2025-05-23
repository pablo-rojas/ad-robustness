#!/usr/bin/env python3
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm

# Import dataset and evaluation utilities.
from src.dataset_utils import get_dataset
from src.detector import UninformedStudents
from src.model_utils import resnet18_classifier
from src.eval_utils import *
from src.misc_utils import load_config, setup_attack_kwargs, get_targeted

# Import ACGAN modules.
from ACGAN.GAN.acgan_1 import ACGAN
from ACGAN.GAN.acgan_res import ACGAN_Res
from ACGAN.attacks.cw import CW
from ACGAN.attacks.FGSM import FGSM

# Import PGD attacker.
from robustness import attacker


def main(config):
    # Directories for models and results.
    results_dir = os.path.join("results", config['experiment_name'])
    n_samples = config['test']['samples']
    ensure_succesful_attack = config['test']['ensure_succesful_attack']

    # Create directories if they do not exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_dir + "/img", exist_ok=True)

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
    #detector = ClassConditionalUninformedStudents(config['num_students'], dataset, patch_size=config['patch_size'], device=device)
    detector.load(config['uninformed_students_path'])
    detector.eval()
    detector.to(device)

    im_channel = 3  # resnet18 expects 3-channel images
    in_dim = 500
    class_dim = 10

    if (config['dataset'] == 'mnist'):
        im_channel = 1
        gan = ACGAN(
            in_dim=50,
            class_dim=10,
            g_filters=[384, 192, 96, 48, im_channel],
            g_strides=[1, 2, 2, 2],
            d_filters=[16, 32, 64, 128, 256, 512],
            d_strides=[2, 1, 2, 1, 2, 1],
            CNN=target_model
        )
    else:
        gan = ACGAN_Res(in_dim=in_dim, class_dim=class_dim, CNN=target_model)

    gan.load_model(config['acgan_path'])
    gan.generator.eval()
    gan.discriminator.eval()

    # Set up attackers.
    pgd_attacker = attacker.Attacker(target_model, dataset).to(device)

    # -------------------------------
    # 1. Natural (clean) evaluation loop.
    # -------------------------------
    nat_as_ACGAN = []  # Will store anomaly scores for clean images.
    nat_as_UninformedStudents = []  # Will store anomaly scores for clean images.
    nat_accuracy = 0

    print("Starting evaluation on natural (clean) images ...")

    i = 0
    pbar = tqdm(total=n_samples, desc="Evalutaing on Natural Images")
    for images, labels in test_loader:
        y = target_model(images.to(device)).detach().cpu()

        # if incorrect prediction, skip the sample
        if ensure_succesful_attack and (y.argmax(1) != labels).sum().item() > 0:
                continue
        
        # Calculate the accuracy on natural images.
        nat_accuracy += (y.argmax(1) == labels).sum().item()/n_samples

        # Calculate AS for ACGAN
        as_ACGAN = sd_statistic(gan.discriminator(images.to(device)), labels)
        nat_as_ACGAN.append(as_ACGAN.item())

        # Calculate AS from Regression Error (ru) and Predictive Uncertanty (pu) for Uninformed Students
        re, pu = detector.forward(norm(images.to(device)), labels)
        as_UninformedStudents = (re - detector.e_mean) / detector.e_std + (pu - detector.v_mean) / detector.v_std
        nat_as_UninformedStudents.append(as_UninformedStudents)

        save_image(config['test']['save'], results_dir + "/img/"+str(i) + "_nat.png", norm(images)[0].detach().cpu(), dataset)

        pbar.update(1)
        i += 1
        if i >= n_samples:
            break

    nat_as_ACGAN = np.array(nat_as_ACGAN)
    nat_as_UninformedStudents = np.array(nat_as_UninformedStudents)

    # Calculate the top 1% quantile 
    threshold_ACGAN = torch.quantile(torch.tensor(nat_as_ACGAN), 0.90).item()
    threshold_UninformedStudents = torch.quantile(torch.tensor(nat_as_UninformedStudents), 0.90).item()

    # -------------------------------s
    # 2. Adversarial evaluation loop.
    # -------------------------------

    # Iterate over the list of attack configurations.
    for attack_config in config['test']['attack_list']:

        # Initialize lists to store the anomaly scores for adversarial images.
        adv_as_ACGAN = []
        adv_as_UninformedStudents = []
        adv_accuracy = 0

        # See if attack is targeted
        targeted, str_targeted = get_targeted(attack_config)

        # Create individual directories for each attack configuration.
        results_dir_attack = results_dir + "/" + attack_config['type'] + "_" + str_targeted + "_" + attack_config['constraint'] + "_" + str(attack_config['epsilon']) 
        os.makedirs(results_dir_attack, exist_ok=True)
        os.makedirs(results_dir_attack + "/acgan", exist_ok=True)
        os.makedirs(results_dir_attack + "/uninformed_students", exist_ok=True)
        os.makedirs(results_dir_attack + "/img", exist_ok=True)

        # Initialize the attacker (or attack args for PGD).
        if (attack_config['type'] == 'pgd'):
             attack_kwargs = setup_attack_kwargs(attack_config)    

        elif (attack_config['type'] == 'cw'):
            cw = CW(model=target_model, norm=norm, kappa=attack_config['kappa'], steps=attack_config['iterations'], lr=attack_config['step_size'], targeted=attack_config['targeted'])

        elif (attack_config['type'] == 'fgsm'):
            fgsm = FGSM(model=target_model, norm=norm, epsilon=attack_config['epsilon'], targeted=attack_config['targeted'])

        l2_dist = []
        linf_dist = []
        i = 0
        pbar = tqdm(total=n_samples, desc="Adversarial Evaluation with " + attack_config['type'] + "_" + attack_config['constraint'] + "_" + str(attack_config['epsilon']))
        for images, labels in test_loader:
            
            # if incorrect prediction, skip the sample
            if ensure_succesful_attack and (target_model(images.to(device)).argmax(1) != labels.to(device)).sum().item() > 0:
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

            # Compute classification accuracy on adversarial images (using the classifier on GPU).
            y = target_model(adv_images.to(device)).detach().cpu()

            # if ensure_succesful_attack and (y.argmax(1) != labels).sum().item() < len(labels):
            #     continue
                    
            # Calculate the accuracy on adversarial images.
            adv_accuracy += (y.argmax(1) == labels).sum().item()/n_samples

            # Calculate L2 and Linf distances between the clean and adversarial images.
            l2_dist.append(torch.norm(adv_images - images.to(device), p=2).item())
            linf_dist.append(torch.norm(adv_images - images.to(device), p=float('inf')).item())

            # Calculate AS for ACGAN
            as_ACGAN = sd_statistic(gan.discriminator(adv_images),target_labels) #y.argmax(1))
            adv_as_ACGAN.append(as_ACGAN.item())

            # Calculate AS for Uninformed Students
            re, pu = detector.forward(norm(adv_images).to(device), y.argmax(1))
            as_UninformedStudents = (re - detector.e_mean) / detector.e_std + (pu - detector.v_mean) / detector.v_std
            adv_as_UninformedStudents.append(as_UninformedStudents)

            save_image(config['test']['save'], results_dir_attack + "/img/"+str(i) + "_adv.png", norm(adv_images)[0].detach().cpu(), dataset)

            pbar.update(1)
            i += 1
            if i >= n_samples:
                break

        # Convert lists to numpy arrays for easier processing.
        adv_as_ACGAN = np.array(adv_as_ACGAN)
        adv_as_UninformedStudents = np.array(adv_as_UninformedStudents)
        l2_dist = np.array(l2_dist)
        linf_dist = np.array(linf_dist)

        # -------------------------------
        # 3. Compute detection metrics.
        # -------------------------------
        # Use the 90th percentile of normalized natural anomaly scores as the threshold.
        detected_ACGAN = np.sum(adv_as_ACGAN > threshold_ACGAN)
        acc_ACGAN = (detected_ACGAN / len(adv_as_ACGAN) * 100) if len(adv_as_ACGAN) > 0 else 0

        detected_UninformedStudents = np.sum(adv_as_UninformedStudents > threshold_UninformedStudents)
        acc_UninformedStudents = (detected_UninformedStudents / len(adv_as_UninformedStudents) * 100) if len(adv_as_UninformedStudents) > 0 else 0

        # Compute partial AUC (pAUC) using the provided utility.
        pAUC_ACGAN = partial_auc(nat_as_ACGAN.tolist(), adv_as_ACGAN.tolist())
        pAUC_UninformedStudents = partial_auc(nat_as_UninformedStudents, adv_as_UninformedStudents.tolist())

        # -------------------------------s
        # 4. Save and display results.
        # -------------------------------
        results_ACGAN = {
            "nat_list": nat_as_ACGAN.tolist(),
            "adv_list": adv_as_ACGAN.tolist(),
            "threshold": threshold_ACGAN,
            "natural_accuracy": nat_accuracy,
            "adversarial_accuracy": adv_accuracy,
            "adversarial_detection_accuracy": acc_ACGAN,
            "pAUC": pAUC_ACGAN,
            "l2_dist": l2_dist,
            "linf_dist": linf_dist
        }
        save_results(results_dir_attack + '/acgan', results_ACGAN, range=(0, 101))
        
        results_UninformedStudents = {
            "nat_list": nat_as_UninformedStudents,
            "adv_list": adv_as_UninformedStudents.tolist(),
            "threshold": threshold_UninformedStudents,
            "natural_accuracy": nat_accuracy,
            "adversarial_accuracy": adv_accuracy,
            "adversarial_detection_accuracy": acc_UninformedStudents,
            "pAUC": pAUC_UninformedStudents,
            "l2_dist": l2_dist,
            "linf_dist": linf_dist
        }
        save_results(results_dir_attack + '/uninformed_students', results_UninformedStudents, range=(-3, 17))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ACGAN with PGD adversarial attack and compute anomaly score metrics")
    parser.add_argument('--config', type=str, default='cfg/cifar_benchmark.json', help='Path to the configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)

    main(config)