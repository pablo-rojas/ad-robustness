import numpy as np
from sklearn.metrics import roc_curve, auc

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
