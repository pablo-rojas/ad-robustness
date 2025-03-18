import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from src.dataset_utils import denormalize_image


def save_results(save_folder, results, range=(-3, 17), bins=100):
    """
    Save evaluation results to disk, including plots and metrics.
    
    Args:
        save_folder (str): Directory where results will be saved
        results (dict): Dictionary containing evaluation results with keys 'nat_list' and 'adv_list'
        score_range (tuple): Range for x-axis in histogram and CDF plots (min, max)
        bins (int): Number of bins for histogram plots
    """
    # Ensure the save directory exists
    os.makedirs(save_folder, exist_ok=True)

    # Save the results dictionary as a numpy file
    np.save(os.path.join(save_folder, "results.npy"), results)

    # Generate and save plots
    save_histogram(save_folder, results['nat_list'], results['adv_list'], range, bins)
    save_cdf(save_folder, results['nat_list'], results['adv_list'], range, bins)
    save_inv_cdf(save_folder, results['nat_list'], results['adv_list'], range, bins)
    save_roc_curve(save_folder, results['nat_list'], results['adv_list'])

    # Save scalar values to results.txt, excluding list values
    with open(os.path.join(save_folder, "results.txt"), "w") as f:
        for key, value in results.items():
            if key not in ['adv_list', 'nat_list']:
                f.write(f"{key}: {value}\n")


def save_image(save, filename, image, dataset):
    """
    Save a tensor image to disk after denormalization.
    
    Args:
        save (bool): Whether to save the image or not
        filename (str): Path where the image will be saved
        image (torch.Tensor): Image tensor in (C, H, W) format
        dataset (str): Dataset name for applying appropriate denormalization
    """
    if save:
        # Denormalize the image
        image = denormalize_image(image, dataset)
            
        # Convert the tensor to numpy array in (H, W, C) format and scale to [0, 255]
        image_np = image.permute(1, 2, 0).numpy()
        image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(filename, image_np)


def save_histogram(save_folder, nat_list, adv_list, score_range=(-3, 17), bins=100):
    """
    Create and save a histogram comparing natural and adversarial anomaly scores.
    
    Args:
        save_folder (str): Directory where the plot will be saved
        nat_list (list): Natural (clean) image anomaly scores
        adv_list (list): Adversarial image anomaly scores
        score_range (tuple): Range for x-axis (min, max)
        bins (int): Number of bins for histogram
    """
    plt.figure(figsize=(10, 6))
    plt.hist(nat_list, bins=bins, range=score_range, color='green', alpha=0.6, label="Natural Anomaly Scores")
    plt.hist(adv_list, bins=bins, range=score_range, color='red', alpha=0.6, label="Adversarial Anomaly Scores")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")
    plt.title("Anomaly Score Distributions")
    plt.xticks(np.arange(score_range[0], score_range[1] + 1, 1))
    plt.savefig(os.path.join(save_folder, "anomaly_score_histogram.png"))
    plt.close()


def save_cdf(save_folder, nat_list, adv_list, score_range=(-3, 17), bins=100):
    """
    Create and save a plot of the cumulative distribution functions (CDF) for anomaly scores.
    
    Args:
        save_folder (str): Directory where the plot will be saved
        nat_list (list): Natural (clean) image anomaly scores
        adv_list (list): Adversarial image anomaly scores
        score_range (tuple): Range for x-axis (min, max)
        bins (int): Number of bins for CDF calculation
    """
    plt.figure(figsize=(10, 6))
    plt.hist(nat_list, bins=bins, range=score_range, color='green', alpha=0.6, 
             label="Natural Anomaly Scores", cumulative=True, density=True, histtype='step')
    plt.hist(adv_list, bins=bins, range=score_range, color='red', alpha=0.6, 
             label="Adversarial Anomaly Scores", cumulative=True, density=True, histtype='step')
    plt.xlabel("Anomaly Score")
    plt.ylabel("CDF")
    plt.legend(loc="lower right")
    plt.title("Anomaly Score CDFs")
    plt.xticks(np.arange(score_range[0], score_range[1] + 1, 1))
    plt.savefig(os.path.join(save_folder, "anomaly_score_cdf.png"))
    plt.close()


def save_inv_cdf(save_folder, nat_list, adv_list, score_range=(-3, 17), bins=100):
    """
    Create and save a plot comparing inverse CDF for natural scores and CDF for adversarial scores.
    
    Args:
        save_folder (str): Directory where the plot will be saved
        nat_list (list): Natural (clean) image anomaly scores
        adv_list (list): Adversarial image anomaly scores
        score_range (tuple): Range for x-axis (min, max)
        bins (int): Number of bins for CDF calculation
    """
    plt.figure(figsize=(10, 6))
    # Plot the CDF for natural anomaly scores (inverted)
    nat_hist, nat_bins = np.histogram(nat_list, bins=bins, range=score_range, density=True)
    nat_cdf = np.cumsum(nat_hist) / np.sum(nat_hist)
    plt.plot(nat_bins[1:], 1 - nat_cdf, color='green', alpha=0.6, label="Natural Anomaly Scores (Inverted CDF)")
    
    # Plot the CDF for adversarial anomaly scores without inversion
    adv_hist, adv_bins = np.histogram(adv_list, bins=bins, range=score_range, density=True)
    adv_cdf = np.cumsum(adv_hist) / np.sum(adv_hist)
    plt.plot(adv_bins[1:], adv_cdf, color='red', alpha=0.6, label="Adversarial Anomaly Scores")
    
    plt.xlabel("Anomaly Score")
    plt.ylabel("CDF")
    plt.legend(loc="lower right")
    plt.title("Anomaly Score CDFs")
    plt.xticks(np.arange(score_range[0], score_range[1] + 1, 1))
    plt.savefig(os.path.join(save_folder, "anomaly_score_inv_cdf.png"))
    plt.close()


def save_roc_curve(save_folder, nat_list, adv_list):
    """
    Create and save a ROC curve plot for the anomaly detection performance.
    
    Args:
        save_folder (str): Directory where the plot will be saved
        nat_list (list): Natural (clean) image anomaly scores
        adv_list (list): Adversarial image anomaly scores
    """
    # Ensure inputs are numpy arrays and combine scores
    anomaly_free_scores = np.array(nat_list)
    anomalous_scores = np.array(adv_list)
    y_scores = np.concatenate([anomaly_free_scores, anomalous_scores])
    y_true = np.concatenate([np.zeros(len(anomaly_free_scores)), np.ones(len(anomalous_scores))])

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
    
    Args:
        anomaly_free_scores (list or np.array): Anomaly scores for normal images
        anomalous_scores (list or np.array): Anomaly scores for anomalous images
        fpr_threshold (float): Maximum false positive rate to consider for the partial AUC
    
    Returns:
        float: The partial AUC value up to the specified FPR threshold
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
    
    Args:
        labels (torch.Tensor): Tensor containing the true labels
        
    Returns:
        torch.Tensor: A randomly selected label different from the input label
    """
    a = random.randint(0, 9)
    while a == labels[0]:
        a = random.randint(0, 9)
    return torch.tensor([a])


def sd_statistic(discriminator_output, target_label):
    """
    Converts discriminator output into a single statistic for anomaly detection.
    
    The function changes the sign of the output to be positive and converts 
    all negative results to 100 (a very large number).

    Args:
        discriminator_output (tuple): Tuple containing (probability tensor, output tensor)
        target_label (int): Target label of the attack
        
    Returns:
        torch.Tensor: The anomaly detection statistic
    """
    aux_prob, aux_out = discriminator_output
    s_d = torch.log(aux_prob) + torch.log(aux_out[:, target_label])
    if torch.isneginf(s_d).any():
        return torch.tensor(-100.0, device=s_d.device)
    return -s_d