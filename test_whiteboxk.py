#!/usr/bin/env python3
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
# import torchvision.models as models
from robustness.cifar_models import resnet50

# Import dataset and evaluation utilities.
from src.dataset_utils import get_dataset
from src.detector import UninformedStudents
from src.model_utils import resnet18_classifier
from src.eval_utils import *
from src.misc_utils import load_config
from src.attacks import PGDWhiteBox


def main(config):
    # Directories for models and results.
    n_samples = 100
    epsilon = 0.1
    num_steps = 100
    lr = 2.5 * epsilon / num_steps  # Step size for PGD attack
    robust_target = False  # Set to True if you want to use a robust target model

    # List of k values to sweep
    ks = [0.0, 0.1, 1.0, 10.0]

    # Use GPU for the classification network and PGD attacker.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset and create a per-sample loader.
    dataset = get_dataset(config['dataset'])
    test_loader = dataset.make_loaders(workers=0, batch_size=1, only_test=True)
    norm = dataset.normalize

    # Load the pretrained ResNet18 classifier.
    # target_model = resnet18_classifier(device, config['dataset'], config['target_model_path'])

    target_model = resnet50()
    target_model.to(device)

    if robust_target:
        ckpt = torch.load('models/cifar_linf_8.pt', map_location=device)

        raw_sd = ckpt['state_dict']

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
    else:
        state_dict = torch.load('models/cifar_nat.pt', map_location=device)
    
    target_model.load_state_dict(state_dict, strict=False)
    target_model.eval()
    target_model.to(device)

    # Initialize Uninformed Students detector
    detector = UninformedStudents(config, device=device)
    detector.load(config['uninformed_students_path'])
    detector.eval()
    detector.to(device)

    # Pre-create PGDWhiteBox attackers for each k
    pgd_wb_attackers = {
        k: PGDWhiteBox(
            model=target_model,
            norm=norm,
            device=device,
            detector=detector,
            k=k,
            epsilon=epsilon,
            alpha=lr,
            num_iter=num_steps
        )
        for k in ks
    }

    # Containers for results
    nat_as = []
    nat_accuracy = []
    adv_wb_as = {k: [] for k in ks}
    adv_wb_accuracy = {k: [] for k in ks}

    print("Starting evaluation ...")
    pbar = tqdm(total=n_samples, desc="Evaluating")
    for i, (images, labels) in enumerate(test_loader):
        if i >= n_samples:
            break

        # Natural images evaluation
        logits = target_model(norm(images.to(device))).detach().cpu()
        nat_accuracy.append((logits.argmax(1) == labels).item())
        re_nat, pu_nat = detector.forward(norm(images.to(device)))
        AS_nat = (re_nat - detector.e_mean) / detector.e_std + (pu_nat - detector.v_mean) / detector.v_std
        nat_as.append(AS_nat)

        # Adversarial images evaluation for each k
        for k, attacker in pgd_wb_attackers.items():
            adv_images = attacker.attack(images.to(device), get_target(labels).to(device))
            logits_adv = target_model(norm(adv_images).to(device)).detach().cpu()
            adv_wb_accuracy[k].append((logits_adv.argmax(1) == labels).item())
            re_adv, pu_adv = detector.forward(norm(adv_images.to(device)))
            AS_adv = (re_adv - detector.e_mean) / detector.e_std + (pu_adv - detector.v_mean) / detector.v_std
            adv_wb_as[k].append(AS_adv)

        pbar.update(1)
    pbar.close()

    # Convert to numpy
    nat_as = np.array(nat_as)
    nat_accuracy = np.array(nat_accuracy)
    for k in ks:
        adv_wb_as[k] = np.array(adv_wb_as[k])
        adv_wb_accuracy[k] = np.array(adv_wb_accuracy[k])

    # Compute natural classifier accuracy
    classifier_nat_acc = nat_accuracy.mean() * 100

    # Compute 10% and 1% FPR thresholds on natural AS
    th10 = torch.quantile(torch.tensor(nat_as), 0.90).item()
    th1 = torch.quantile(torch.tensor(nat_as), 0.99).item()

    # Prepare table rows
    headers = [
        "k",
        "Classifier Acc",
        "Detector Acc (10% FPR)",
        "Classifier Acc Without detected attacks",
        "Detector Acc (1% FPR)",
        "Classifier Acc Without detected attacks",
    ]
    rows = []

    rows.append([
    "Nat",
    f"{classifier_nat_acc:.1f}",
    f"{(nat_as <= th10).sum() / len(nat_as) * 100:.1f}",
    f"{nat_accuracy[nat_as <= th10].mean() * 100:.1f}",
    f"{(nat_as <= th1).sum() / len(nat_as) * 100:.1f}",
    f"{nat_accuracy[nat_as <= th1].mean() * 100:.1f}",
    ])

    for k in ks:
        as_k = adv_wb_as[k]
        acc_k = adv_wb_accuracy[k]

        # Detector decisions
        det10 = (as_k > th10).sum() / len(as_k) * 100
        keep10 = acc_k[as_k <= th10]
        cls_keep10 = keep10.mean() * 100 if keep10.size > 0 else 0.0

        det1 = (as_k > th1).sum() / len(as_k) * 100
        keep1 = acc_k[as_k <= th1]
        cls_keep1 = keep1.mean() * 100 if keep1.size > 0 else 0.0

        rows.append([
            f"{k}",
            f"{acc_k.mean() * 100:.1f}",
            f"{det10:.1f}",
            f"{cls_keep10:.1f}",
            f"{det1:.1f}",
            f"{cls_keep1:.1f}",
        ])

    # Print table
    print("\n\nResults across different k values:\n")
    print(tabulate(rows, headers=headers, floatfmt=".1f"))


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
