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
    plt.figure(figsize=(10, 6))
    plt.hist(results['nat_list'], bins=bins, range=range, color='green', alpha=0.6, label="Natural Anomaly Scores")
    plt.hist(results['adv_list'], bins=bins, range=range, color='red', alpha=0.6, label="Adversarial Anomaly Scores")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")
    plt.title("Anomaly Score Distributions")
    plt.xticks(np.arange(range[0], range[1] + 1, 1))  # Adding proper x-axis ticks using the range values
    plt.savefig(os.path.join(save_folder, "anomaly_score_histogram.png"))
    plt.close()

    # Save scalar values to results.txt
    with open(os.path.join(save_folder, "results.txt"), "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

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
