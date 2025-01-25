import numpy as np
from sklearn.metrics import roc_curve, auc

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

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

    # Save scalar values to results.txt
    with open(os.path.join(save_folder, "results.txt"), "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

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
    # Combine the scores and create corresponding labels
    y_true = np.array([0] * len(anomaly_free_scores) + [1] * len(anomalous_scores))
    y_scores = np.array(anomaly_free_scores + anomalous_scores)
    
    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    # Find the points where the FPR is below the threshold
    fpr_threshold_indices = np.where(fpr <= fpr_threshold)[0]
    
    # Restrict the FPR and TPR to those points
    fpr_restricted = fpr[fpr_threshold_indices]
    tpr_restricted = tpr[fpr_threshold_indices]
    
    # Calculate the partial AUC using trapezoidal integration
    pAUC = auc(fpr_restricted, tpr_restricted)
    
    return pAUC
