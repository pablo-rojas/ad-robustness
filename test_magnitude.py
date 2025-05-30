#!/usr/bin/env python3
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# import torchvision.models as models
from robustness.cifar_models import resnet50, resnet18

# Import dataset and evaluation utilities.
from src.dataset_utils import get_dataset
from src.detector import UninformedStudents
from src.model_utils import resnet18_classifier
from src.eval_utils import *
from src.misc_utils import load_config
from src.attacks import *

from ACGAN.GAN.acgan_res import ACGAN_Res


def main(config):
    # Directories for models and results.
    n_samples = 500
    robust_target = False  # True: robust resnet50, False: non-robust resnet50, None: non-robust resnet18
    targeted = -1  # Set to True if you want to use targeted attacks
    only_success = True  # Set to True if you want to only evaluate successful attacks
    detector_type = 'US'  # Detector type: 'US' for Uninformed Students, 'ACGAN' for ACGAN
    loss_fn = torch.nn.CrossEntropyLoss()

    epsilon_list = [ i/255.0 for i in range(100) ]  # Epsilon values for attack

    # Use GPU for the classification network and PGD attacker.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset and create a per-sample loader.
    dataset = get_dataset(config['dataset'])
    test_loader = dataset.make_loaders(workers=0, batch_size=1, only_test=True)
    norm = dataset.normalize

    target_model = resnet50()
    target_model.to(device)

    target_model = initialize_target_model(config, robust_target, device)

    # Initialize Uninformed Students detector
    if detector_type == 'US':
        detector = UninformedStudents(config, device=device)
        detector.load(config['uninformed_students_path'])
        detector.eval()
        detector.to(device)

    elif detector_type == 'ACGAN':
        detector = ACGAN_Res(in_dim=500, class_dim=10, CNN=target_model)
        detector.load_model(config['acgan_path'])
        detector.generator.eval()
        detector.discriminator.eval()
    else:
        raise ValueError("Invalid detector type. Choose 'US' or 'ACGAN'.")

    # Containers for results
    as_list = {eps: [] for eps in epsilon_list}
    acc = {eps: [] for eps in epsilon_list}

    i = 0
    print("Starting evaluation ...")
    pbar = tqdm(total=n_samples, desc="Evaluating")
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        images.requires_grad = True
        if i >= n_samples:
            break

        # Get target labels for the attack
        target_labels = get_target(labels).to(device) if targeted else labels.to(device)

        # FGSM attack
        output = target_model(norm(images))
        loss = -1 * targeted * loss_fn(output, target_labels)
        loss.backward()
        images_grad = images.grad.data
        sign_data_grad = images_grad.sign()

        for eps in epsilon_list:
            adv_images = images + eps * sign_data_grad
            adv_images = torch.clamp(adv_images, 0, 1)

            with torch.no_grad():
                logits = target_model(norm(adv_images).to(device)).detach()
                acc[eps].append((logits.argmax(1) == labels).item())

                if detector_type == 'US':
                    re, pu = detector.forward(norm(adv_images.to(device)))
                    AS = (re - detector.e_mean) / detector.e_std + (pu - detector.v_mean) / detector.v_std
                elif detector_type == 'ACGAN':
                    AS = sd_statistic(detector.discriminator(adv_images.to(device)), logits.argmax(1))

                as_list[eps].append(AS.item())

        pbar.update(1)
        i += 1
    pbar.close()

    # Convert results to numpy arrays
    for eps in epsilon_list:
        as_list[eps] = np.array(as_list[eps])
        acc[eps] = np.array(acc[eps])

    # Check if the classifier was incorrect on nat images
    if only_success:
        filter = acc[0] == 1

        for eps in epsilon_list[1:]:
            as_list[eps] = as_list[eps][filter]
            #acc[eps] = acc[eps][filter]
    
    mean_as = [as_list[eps].mean() for eps in epsilon_list]
    std_as = [as_list[eps].std() for eps in epsilon_list]
    mean_acc = [acc[eps].mean()*100 for eps in epsilon_list]

    # Calculate thresholds for 10% and 1% false positive rates on natural images
    th10 = torch.quantile(torch.tensor(as_list[epsilon_list[0]]), 0.90).item()
    th1 = torch.quantile(torch.tensor(as_list[epsilon_list[0]]), 0.99).item()

    # Plot the results
    # Plot the results (combined with dual y-axis)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Anomaly score on left axis (log scale)
    ax1.plot(epsilon_list, mean_as, label='Anomaly Score (mean)', linestyle='-')
    ax1.fill_between(
        epsilon_list,
        [m - s for m, s in zip(mean_as, std_as)],
        [m + s for m, s in zip(mean_as, std_as)],
        alpha=0.3,
        label='Anomaly Score (std)',
    )
        # FPR thresholds at ε = 0
    ax1.axhline(
        th10,
        linestyle='--',
        label='10% FPR threshold'
    )
    ax1.axhline(
        th1,
        linestyle=':',
        label='1% FPR threshold'
    )
    ax1.set_xscale('linear')
    ax1.set_yscale('log')
    ax1.set_xlabel('ε (attack strength)')
    ax1.set_ylabel('Anomaly Score (log scale)')
    ax1.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

    # Accuracy on right axis (linear scale)
    ax2.plot(epsilon_list, mean_acc, color='C1', label='Classifier Accuracy (%)', linestyle='-')
    ax2.set_ylabel('Classifier Accuracy (%)')

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

    # plt.title('Anomaly Score and Classifier Accuracy vs. ε')
    fig.tight_layout()
    plt.savefig('combined_plot.png')

    

def initialize_target_model(config, robust_target, device):
    if robust_target:
        target_model = resnet18()

        ckpt = torch.load('models/resnet18_cifar_rob.pt', map_location=device)

        raw_sd = ckpt['model']
        
        fixed_sd = {}
        for k, v in raw_sd.items():
            name = k
            if name.startswith('module.'):
                name = name[len('module.'):]
            if name.startswith('model.'):
                name = name[len('model.'):]
            fixed_sd[name] = v
        model_keys = set(target_model.state_dict().keys())
        state_dict = {k: v for k, v in fixed_sd.items() if k in model_keys}
        target_model.load_state_dict(state_dict)
    else:
        target_model = resnet18_classifier(device, config['dataset'], config['target_model_path'])

        # ckpt = torch.load('models/cifar_nat.pt', map_location=device)
        # raw_sd = ckpt['model']
        
        # fixed_sd = {}
        # for k, v in raw_sd.items():
        #     name = k
        #     if name.startswith('module.'):
        #         name = name[len('module.'):]
        #     if name.startswith('model.'):
        #         name = name[len('model.'):]
        #     fixed_sd[name] = v
        # model_keys = set(target_model.state_dict().keys())
        # state_dict = {k: v for k, v in fixed_sd.items() if k in model_keys}
        # target_model.load_state_dict(state_dict)
    
    
    target_model.eval()
    target_model.to(device)
    return target_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PGDWhiteBox over multiple k values"
    )
    parser.add_argument(
        '--config', type=str, default='cfg/cifar_benchmark.json',
        help='Path to the configuration file.'
    )
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
