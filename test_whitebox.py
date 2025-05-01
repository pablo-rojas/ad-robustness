#!/usr/bin/env python3
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
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
    n_samples = 100
    epsilon = 0.1  # Epsilon for PGD attack
    num_steps = 20
    step_size = 2.5 * epsilon / num_steps  # Step size for PGD attack
    robust_target = True  # True: robust resnet50, False: non-robust resnet50, None: non-robust resnet18
    targeted = True  # Set to True if you want to use targeted attacks
    only_success = False  # Set to True if you want to only evaluate successful attacks
    detector_type = 'US'  # Detector type: 'US' for Uninformed Students, 'ACGAN' for ACGAN
    normalize_grad = False # Set to True if you want to normalize the gradients during the attack

    # List of k values to sweep
    #ks = [0.0, 1.0, 100.0, 10000.0]
    ks = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    # ks = [0.0]
    # ks =[1000.0]

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

    target_model = initialize_target_model(config, robust_target, device)

    # Initialize Uninformed Students detector
    if detector_type == 'US':
        detector = UninformedStudents(config, device=device)
        detector.load(config['uninformed_students_path'])
        detector.eval()
        detector.to(device)

        # Pre-create PGDWhiteBox attackers for each k
        pgd_wb_attackers = {
            k: PGDus(
                model=target_model,
                norm=norm,
                device=device,
                detector=detector,
                k=k,
                epsilon=epsilon,
                step_size=step_size,
                num_iter=num_steps,
                targeted=targeted,
                norm_grad=normalize_grad,
            )
            for k in ks
        }
    elif detector_type == 'ACGAN':
        detector = ACGAN_Res(in_dim=500, class_dim=10, CNN=target_model)
        detector.load_model(config['acgan_path'])
        detector.generator.eval()
        detector.discriminator.eval()


        # Pre-create PGDWhiteBox attackers for each k
        pgd_wb_attackers = {
            k: PGDacgan(
                model=target_model,
                norm=norm,
                device=device,
                detector=detector,
                k=k,
                epsilon=epsilon,
                step_size=step_size,
                num_iter=num_steps,
                targeted=targeted,
                norm_grad=normalize_grad,
            )
            for k in ks
        }
    else:
        raise ValueError("Invalid detector type. Choose 'US' or 'ACGAN'.")

    # Containers for results
    nat_as = []
    nat_accuracy = []
    adv_wb_as = {k: [] for k in ks}
    adv_wb_accuracy = {k: [] for k in ks}
    att_success = {k: [] for k in ks}
    pAUC = {k: [] for k in ks}

    i = 0
    print("Starting evaluation ...")
    pbar = tqdm(total=n_samples, desc="Evaluating")
    for images, labels in test_loader:
        if i >= n_samples:
            break

        with torch.no_grad():
            # Natural images evaluation
            logits = target_model(norm(images.to(device))).detach().cpu()

            # Ensure successful attack
            if only_success and (logits.argmax(1) != labels).item():
                continue

            nat_accuracy.append((logits.argmax(1) == labels).item())
            if detector_type == 'US':
                re_nat, pu_nat = detector.forward(norm(images.to(device)))
                AS_nat = (re_nat - detector.e_mean) / detector.e_std + (pu_nat - detector.v_mean) / detector.v_std
            elif detector_type == 'ACGAN':
                AS_nat = sd_statistic(detector.discriminator(images.to(device)), logits.argmax(1))
            nat_as.append(AS_nat.item())

        # Get target labels for the attack
        target_labels = get_target(labels).to(device) if targeted else labels.to(device)

        # Adversarial images evaluation for each k
        for k, attacker in pgd_wb_attackers.items():
            adv_images = attacker.attack(images.to(device), target_labels)
            with torch.no_grad():
                logits_adv = target_model(norm(adv_images).to(device)).detach().cpu()
                adv_wb_accuracy[k].append((logits_adv.argmax(1) == labels).item())

                if detector_type == 'US':
                    re_adv, pu_adv = detector.forward(norm(adv_images.to(device)))
                    AS_adv = (re_adv - detector.e_mean) / detector.e_std + (pu_adv - detector.v_mean) / detector.v_std
                elif detector_type == 'ACGAN':
                    AS_adv = sd_statistic(detector.discriminator(adv_images.to(device)), logits_adv.argmax(1))

                adv_wb_as[k].append(AS_adv.item())
                if targeted:
                    att_success[k].append((logits_adv.argmax(1) == target_labels.cpu()).item())
                else:
                    att_success[k].append((logits_adv.argmax(1) != target_labels.cpu()).item())

        pbar.update(1)
        i += 1
    pbar.close()

    # Convert to numpy
    nat_as = np.array(nat_as)
    nat_accuracy = np.array(nat_accuracy)
    for k in ks:
        adv_wb_as[k] = np.array(adv_wb_as[k])
        adv_wb_accuracy[k] = np.array(adv_wb_accuracy[k])
        att_success[k] = np.array(att_success[k])
        pAUC[k] = partial_auc(nat_as.tolist(), adv_wb_as[k].tolist())

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
        "Detector Acc (1% FPR)",
        "pAUC-0.2",
        "Attack Success Rate"
    ]
    rows = []

    rows.append([
        "Nat",
        f"{classifier_nat_acc:.1f}",
        f"{(nat_as <= th10).sum() / len(nat_as) * 100:.1f}",
        f"{(nat_as <= th1).sum() / len(nat_as) * 100:.1f}",
        "",
        ""
    ])

    for k in ks:
        as_k = adv_wb_as[k]
        acc_k = adv_wb_accuracy[k]
        succ_k = att_success[k]  # original per-sample adv success (bool array)

        # Detector decision thresholds
        det10 = (as_k > th10).sum() / len(as_k) * 100
        det1 = (as_k > th1).sum() / len(as_k) * 100

        # attack-success definition: both classifier fooled AND not detected at 10% FPR.
        # Evaluated only on samples that were classified correctly by the classifier on
        # natural images.
        atk_succ_rate = np.logical_and(succ_k[nat_accuracy], as_k[nat_accuracy] <= th10).mean() * 100
        # Note how how filtering the results with nat_accuracy already reduces the array
        # size to the number of samples that were classified correctly by the classifier on
        # natural images. This is important because we want to compute the attack success
        # rate only on those samples.

        # success_and = np.logical_and(succ_k, as_k <= th10)
        # atk_succ_rate = success_and.mean() * 100

        # success_and = np.logical_and(succ_k[nat_accuracy], as_k[nat_accuracy] <= th10)

        rows.append([
            f"{k}",
            f"{acc_k.mean() * 100:.1f}",
            f"{det10:.1f}",
            f"{det1:.1f}",
            f"{pAUC[k]:.4f}",
            f"{atk_succ_rate:.1f}",
        ])

    print("\n\nResults across different k values for a PGD attack on " + ('robust' if robust_target else 'non-robust') + " target model and "   + detector_type + " detector:\n")
    print(tabulate(rows, headers=headers, floatfmt=".1f"))

    return headers, rows

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
