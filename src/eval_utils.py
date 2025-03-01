import numpy as np
from sklearn.metrics import roc_curve, auc

import os
import cv2
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from src.dataset_utils import denormalize_image

def save_results(save_folder, results, range=(-3, 17), bins=100):    
    # Ensure the save directory exists
    os.makedirs(save_folder, exist_ok=True)

    # Save the results dictionary as a numpy file
    np.save(os.path.join(save_folder, "results.npy"), results)

    # Plot and save histogram
    save_histogram(save_folder, results['nat_list'], results['adv_list'], range=range, bins=bins)

    # Plot and save CDF
    save_cdf(save_folder, results['nat_list'], results['adv_list'], range=range, bins=bins)
    save_inv_cdf(save_folder, results['nat_list'], results['adv_list'], range=range, bins=bins)


    # Plot and save ROC curve
    save_roc_curve(save_folder, results['nat_list'], results['adv_list'])

    # Save scalar values to results.txt, excluding list values
    with open(os.path.join(save_folder, "results.txt"), "w") as f:
        for key, value in results.items():
            if key not in ['adv_list', 'nat_list']:
                f.write(f"{key}: {value}\n")

def save_image(save, filename, image, dataset):
    if save:
        # Denormalize the first input image in the batch
        image = denormalize_image(image, dataset)  # Apply denormalization
            
        # Convert the tensor back to (H, W, C) format and scale to [0, 255]
        image_np = image.permute(1, 2, 0).numpy()  # Convert to (H, W, C) format
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)  # Scale and convert to uint8

        # convert to BGR
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(filename, image_np)

def save_histogram(save_folder, nat_list, adv_list, range=(-3, 17), bins=100):
    plt.figure(figsize=(10, 6))
    plt.hist(nat_list, bins=bins, range=range, color='green', alpha=0.6, label="Natural Anomaly Scores")
    plt.hist(adv_list, bins=bins, range=range, color='red', alpha=0.6, label="Adversarial Anomaly Scores")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")
    plt.title("Anomaly Score Distributions")
    plt.xticks(np.arange(range[0], range[1] + 1, 1))  # Adding proper x-axis ticks using the range values
    plt.savefig(os.path.join(save_folder, "anomaly_score_histogram.png"))
    plt.close()

def save_cdf(save_folder, nat_list, adv_list, range=(-3, 17), bins=100):
    plt.figure(figsize=(10, 6))
    plt.hist(nat_list, bins=bins, range=range, color='green', alpha=0.6, label="Natural Anomaly Scores", cumulative=True, density=True, histtype='step')
    plt.hist(adv_list, bins=bins, range=range, color='red', alpha=0.6, label="Adversarial Anomaly Scores", cumulative=True, density=True, histtype='step')
    plt.xlabel("Anomaly Score")
    plt.ylabel("CDF")
    plt.legend(loc="lower right")
    plt.title("Anomaly Score CDFs")
    plt.xticks(np.arange(range[0], range[1] + 1, 1))  # Adding proper x-axis ticks using the range values
    plt.savefig(os.path.join(save_folder, "anomaly_score_cdf.png"))
    plt.close()

def save_inv_cdf(save_folder, nat_list, adv_list, range=(-3, 17), bins=100):
    plt.figure(figsize=(10, 6))
    # Plot the CDF for natural anomaly scores (inverted)
    nat_hist, nat_bins = np.histogram(nat_list, bins=bins, range=range, density=True)
    nat_cdf = np.cumsum(nat_hist) / np.sum(nat_hist)
    plt.plot(nat_bins[1:], 1 - nat_cdf, color='green', alpha=0.6, label="Natural Anomaly Scores (Inverted CDF)")
    
    # Plot the CDF for adversarial anomaly scores without inversion
    adv_hist, adv_bins = np.histogram(adv_list, bins=bins, range=range, density=True)
    adv_cdf = np.cumsum(adv_hist) / np.sum(adv_hist)
    plt.plot(adv_bins[1:], adv_cdf, color='red', alpha=0.6, label="Adversarial Anomaly Scores")
    
    plt.xlabel("Anomaly Score")
    plt.ylabel("CDF")
    plt.legend(loc="lower right")
    plt.title("Anomaly Score CDFs")
    plt.xticks(np.arange(range[0], range[1] + 1, 1))  # Adding proper x-axis ticks using the range values
    plt.savefig(os.path.join(save_folder, "anomaly_score_inv_cdf.png"))
    plt.close()

def save_roc_curve(save_folder, nat_list, adv_list):
    # Combine the scores and create corresponding labels
    y_true = np.array([0] * len(nat_list) + [1] * len(adv_list))
    y_scores = np.array(nat_list + adv_list)
    
    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Calculate the AUC
    auc_score = auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_folder, "roc_curve.png"))
    plt.close()


def partial_auc(anomaly_free_scores, anomalous_scores, fpr_threshold=0.2):
    """
    Calculate the partial AUC up to a given false positive rate (FPR) threshold.
    
    Parameters:
    - anomaly_free_scores (list or np.array): Anomaly scores for anomaly-free (normal) images.
    - anomalous_scores (list or np.array): Anomaly scores for anomalous images.
    - fpr_threshold (float): The maximum false positive rate to consider for the partial AUC.
    
    Returns:
    - pAUC (float): The partial AUC value up to the specified FPR threshold.
    """
    # Ensure inputs are numpy arrays and combine scores
    anomaly_free_scores = np.array(anomaly_free_scores)
    anomalous_scores = np.array(anomalous_scores)
    y_scores = np.concatenate([anomaly_free_scores, anomalous_scores])
    y_true = np.concatenate([np.zeros(len(anomaly_free_scores)), np.ones(len(anomalous_scores))])
    
    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Select points where FPR is below the threshold
    valid_idx = np.where(fpr <= fpr_threshold)[0]
    fpr_restricted = fpr[valid_idx]
    tpr_restricted = tpr[valid_idx]
    
    # If the last point is below fpr_threshold, add the point at fpr_threshold via linear interpolation
    if fpr_restricted[-1] < fpr_threshold:
        # Find index of the first FPR value above the threshold
        idx = np.searchsorted(fpr, fpr_threshold)
        # Ensure we don't go out of bounds
        if idx < len(fpr):
            # Linear interpolation for TPR at fpr_threshold
            fpr_low, fpr_high = fpr[idx - 1], fpr[idx]
            tpr_low, tpr_high = tpr[idx - 1], tpr[idx]
            tpr_interp = tpr_low + (tpr_high - tpr_low) * ((fpr_threshold - fpr_low) / (fpr_high - fpr_low))
            
            # Append the interpolated point
            fpr_restricted = np.append(fpr_restricted, fpr_threshold)
            tpr_restricted = np.append(tpr_restricted, tpr_interp)
    
    # Compute the partial AUC using trapezoidal integration
    pAUC = auc(fpr_restricted, tpr_restricted)
    
    return pAUC

def get_target(labels):
    """
    Choose a target class different from the true label.
    """
    a = random.randint(0, 9)
    while a == labels[0]:
        a = random.randint(0, 9)
    return torch.tensor([a])

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def setup_attack_kwargs(config):
    """
    Set up PGD attack parameters based on the configuration.
    """
    if config['type'] == 'pgd':
        attack_kwargs = {
            'constraint': config['constraint'],
            'eps': config['epsilon'],
            'step_size': 2.5 * config['epsilon'] / config['iterations'], #config['step_size'],    # Modification to imitate original PGD paper
            'iterations': config['iterations'],
            'targeted': config['targeted'],
            'custom_loss': None
        }
    return attack_kwargs